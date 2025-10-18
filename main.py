import json
import time
import datetime
from typing import Dict
from dataclasses import dataclass

import astrbot.api.star as star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api import logger


@dataclass
class JudgeResult:
    """判断结果数据类"""
    relevance: float = 0.0
    willingness: float = 0.0
    social: float = 0.0
    timing: float = 0.0
    continuity: float = 0.0  # 新增：与上次回复的连贯性
    reasoning: str = ""
    should_reply: bool = False
    confidence: float = 0.0
    overall_score: float = 0.0
    related_messages: list = None

    def __post_init__(self):
        if self.related_messages is None:
            self.related_messages = []


@dataclass
class ChatState:
    """群聊状态数据类"""
    energy: float = 1.0
    last_reply_time: float = 0.0
    last_reset_date: str = ""
    total_messages: int = 0
    total_replies: int = 0



class HeartflowPlugin(star.Star):

    def __init__(self, context: star.Context, config):
        super().__init__(context)
        self.config = config

        # 判断模型配置
        self.judge_provider_name = self.config.get("judge_provider_name", "")

        # 心流参数配置
        self.reply_threshold = self.config.get("reply_threshold", 0.6)
        self.energy_decay_rate = self.config.get("energy_decay_rate", 0.1)
        self.energy_recovery_rate = self.config.get("energy_recovery_rate", 0.02)
        self.context_messages_count = self.config.get("context_messages_count", 5)
        self.whitelist_enabled = self.config.get("whitelist_enabled", False)
        self.chat_whitelist = self.config.get("chat_whitelist", [])

        # 群聊状态管理
        self.chat_states: Dict[str, ChatState] = {}
        
        # 系统提示词缓存：{conversation_id: {"original": str, "summarized": str, "persona_id": str}}
        self.system_prompt_cache: Dict[str, Dict[str, str]] = {}
        
        # ===== 消息历史缓冲机制 =====
        # 用于保存完整的消息历史，包括未回复的消息
        # 结构：{chat_id: [{"role": str, "content": str, "timestamp": float}]}
        # 
        # 为什么需要这个缓冲区？
        # - AstrBot的conversation_manager只保存被回复的消息
        # - 未回复的消息不会进入对话历史，导致判断时信息缺失
        # - 通过自建缓冲区，确保小模型能看到完整的群聊历史
        #
        # 工作原理：
        # 1. 用户消息：立即记录到缓冲区
        # 2. 机器人回复：从conversation_manager同步到缓冲区
        # 3. 判断时：使用缓冲区的完整历史
        #
        # 注意：缓冲区采用"从现在开始记录"策略，不回溯历史
        self.message_buffer: Dict[str, list] = {}
        self.max_buffer_size = self.config.get("max_buffer_size", 50)  # 每个群聊最多缓存50条

        # 判断配置
        self.judge_include_reasoning = self.config.get("judge_include_reasoning", True)
        self.judge_max_retries = max(0, self.config.get("judge_max_retries", 3))  # 确保最小为0
        
        # 提示词配置
        self.judge_prompt_template = self.config.get("judge_prompt_template", "")
        self.summarize_prompt_template = self.config.get("summarize_prompt_template", "")
        
        # 判断权重配置
        self.weights = {
            "relevance": self.config.get("judge_relevance", 0.25),
            "willingness": self.config.get("judge_willingness", 0.2),
            "social": self.config.get("judge_social", 0.2),
            "timing": self.config.get("judge_timing", 0.15),
            "continuity": self.config.get("judge_continuity", 0.2)
        }
        # 检查权重和
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            logger.warning(f"判断权重和不为1，当前和为{weight_sum}")
            # 进行归一化处理
            self.weights = {k: v / weight_sum for k, v in self.weights.items()}
            logger.info(f"判断权重和已归一化，当前配置为: {self.weights}")

        logger.info("心流插件已初始化")

    async def _get_or_create_summarized_system_prompt(self, event: AstrMessageEvent, original_prompt: str) -> str:
        """获取或创建精简版系统提示词"""
        try:
            # 获取当前会话ID
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(event.unified_msg_origin)
            if not curr_cid:
                return original_prompt
            
            # 获取当前人格ID作为缓存键的一部分
            conversation = await self.context.conversation_manager.get_conversation(event.unified_msg_origin, curr_cid)
            persona_id = conversation.persona_id if conversation else "default"
            
            # 构建缓存键
            cache_key = f"{curr_cid}_{persona_id}"
            
            # 检查缓存
            if cache_key in self.system_prompt_cache:
                cached = self.system_prompt_cache[cache_key]
                # 如果原始提示词没有变化，返回缓存的总结
                if cached.get("original") == original_prompt:
                    logger.debug(f"使用缓存的精简系统提示词: {cache_key}")
                    return cached.get("summarized", original_prompt)
            
            # 如果没有缓存或原始提示词发生变化，进行总结
            if not original_prompt or len(original_prompt.strip()) < 50:
                # 如果原始提示词太短，直接返回
                return original_prompt
            
            summarized_prompt = await self._summarize_system_prompt(original_prompt)
            
            # 更新缓存
            self.system_prompt_cache[cache_key] = {
                "original": original_prompt,
                "summarized": summarized_prompt,
                "persona_id": persona_id
            }
            
            logger.info(f"创建新的精简系统提示词: {cache_key} | 原长度:{len(original_prompt)} -> 新长度:{len(summarized_prompt)}")
            return summarized_prompt
            
        except Exception as e:
            logger.error(f"获取精简系统提示词失败: {e}")
            return original_prompt
    
    async def _summarize_system_prompt(self, original_prompt: str) -> str:
        """使用小模型对系统提示词进行总结"""
        try:
            if not self.judge_provider_name:
                return original_prompt
            
            judge_provider = self.context.get_provider_by_id(self.judge_provider_name)
            if not judge_provider:
                return original_prompt
            
            # 使用配置的提示词模板，如果没有配置则使用默认模板
            if self.summarize_prompt_template:
                summarize_prompt = self.summarize_prompt_template.format(original_prompt=original_prompt)
            else:
                summarize_prompt = f"""请将以下机器人角色设定总结为简洁的核心要点，保留关键的性格特征、行为方式和角色定位。
总结后的内容应该在100-200字以内，突出最重要的角色特点。

原始角色设定：
{original_prompt}

请以JSON格式回复：
{{
    "summarized_persona": "精简后的角色设定，保留核心特征和行为方式"
}}

重要：你的回复必须是完整的JSON对象，不要包含任何其他内容！"""

            llm_response = await judge_provider.text_chat(
                prompt=summarize_prompt,
                contexts=[]  # 不需要上下文
            )

            content = llm_response.completion_text.strip()
            
            # 尝试提取JSON
            try:
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "").strip()
                elif content.startswith("```"):
                    content = content.replace("```", "").strip()

                result_data = json.loads(content)
                summarized = result_data.get("summarized_persona", "")
                
                if summarized and len(summarized.strip()) > 10:
                    return summarized.strip()
                else:
                    logger.warning("小模型返回的总结内容为空或过短")
                    return original_prompt
                    
            except json.JSONDecodeError:
                logger.error(f"小模型总结系统提示词返回非有效JSON: {content}")
                return original_prompt
                
        except Exception as e:
            logger.error(f"总结系统提示词异常: {e}")
            return original_prompt

    async def judge_with_tiny_model(self, event: AstrMessageEvent) -> JudgeResult:
        """使用小模型进行智能判断"""

        if not self.judge_provider_name:
            logger.warning("小参数判断模型提供商名称未配置，跳过心流判断")
            return JudgeResult(should_reply=False, reasoning="提供商未配置")

        # 获取指定的 provider
        try:
            judge_provider = self.context.get_provider_by_id(self.judge_provider_name)
            if not judge_provider:
                logger.warning(f"未找到提供商: {self.judge_provider_name}")
                return JudgeResult(should_reply=False, reasoning=f"提供商不存在: {self.judge_provider_name}")
        except Exception as e:
            logger.error(f"获取提供商失败: {e}")
            return JudgeResult(should_reply=False, reasoning=f"获取提供商失败: {str(e)}")

        # 获取群聊状态
        chat_state = self._get_chat_state(event.unified_msg_origin)

        # 获取当前对话的人格系统提示词，让模型了解大参数LLM的角色设定
        original_persona_prompt = await self._get_persona_system_prompt(event)
        logger.debug(f"小参数模型获取原始人格提示词: {'有' if original_persona_prompt else '无'} | 长度: {len(original_persona_prompt) if original_persona_prompt else 0}")
        
        # 获取或创建精简版系统提示词
        persona_system_prompt = await self._get_or_create_summarized_system_prompt(event, original_persona_prompt)
        logger.debug(f"小参数模型使用精简人格提示词: {'有' if persona_system_prompt else '无'} | 长度: {len(persona_system_prompt) if persona_system_prompt else 0}")

        # 构建判断上下文
        chat_context = await self._build_chat_context(event)

        reasoning_part = ""
        if self.judge_include_reasoning:
            reasoning_part = ',\n    "reasoning": "详细分析原因，说明为什么应该或不应该回复，需要结合机器人角色特点进行分析，特别说明与上次回复的关联性"'

        # 使用配置的提示词模板，如果没有配置则使用默认模板
        if self.judge_prompt_template:
            judge_prompt = self.judge_prompt_template.format(
                persona_system_prompt=persona_system_prompt if persona_system_prompt else "默认角色：智能助手",
                event_unified_msg_origin=event.unified_msg_origin,
                chat_state_energy=chat_state.energy,
                minutes_since_last_reply=self._get_minutes_since_last_reply(event.unified_msg_origin),
                chat_context=chat_context,
                context_messages_count=self.context_messages_count,
                sender_name=event.get_sender_name(),
                message_str=event.message_str,
                current_time=datetime.datetime.now().strftime('%H:%M:%S'),
                reply_threshold=self.reply_threshold,
                reasoning_part=reasoning_part
            )
        else:
            judge_prompt = f"""
你是群聊机器人的决策系统，需要判断是否应该主动回复以下消息。

注意：对话历史已经通过上下文提供给你，你可以从中了解群聊的对话流程。对话历史中：
- [群友消息] 表示群友发送的消息
- [我的回复] 表示机器人（我）发送的回复

机器人角色设定:
{persona_system_prompt if persona_system_prompt else "默认角色：智能助手"}

当前群聊情况:
- 群聊ID: {event.unified_msg_origin}
- 我的精力水平: {chat_state.energy:.1f}/1.0
- 上次发言: {self._get_minutes_since_last_reply(event.unified_msg_origin)}分钟前

群聊基本信息:
{chat_context}

待判断消息:
发送者: {event.get_sender_name()}
内容: {event.message_str}
时间: {datetime.datetime.now().strftime('%H:%M:%S')}

评估要求:
请从以下5个维度评估（0-10分），重要提醒：基于上述机器人角色设定来判断是否适合回复：

1. 内容相关度(0-10)：消息是否有趣、有价值、适合我回复
   - 考虑消息的质量、话题性、是否需要回应
   - 识别并过滤垃圾消息、无意义内容
   - 重要：通过上下文中的对话历史判断这条消息是否是在对我（机器人）说话，还是群友之间的对话
   - 如果这条消息明显是在回复我上次的发言（查看[我的回复]），或者提到我，应该给高分
   - 如果这条消息是群友之间的对话，与我无关，应该给低分
   - 结合机器人角色特点，判断是否符合角色定位

2. 回复意愿(0-10)：基于当前状态，我回复此消息的意愿
   - 考虑当前精力水平和心情状态
   - 考虑今日回复频率控制
   - 基于机器人角色设定，判断是否应该主动参与此话题

3. 社交适宜性(0-10)：在当前群聊氛围下回复是否合适
   - 考虑群聊活跃度和讨论氛围
   - 考虑机器人角色在群中的定位和表现方式

4. 时机恰当性(0-10)：回复时机是否恰当
   - 考虑距离上次回复的时间间隔
   - 考虑消息的紧急性和时效性

5. 对话连贯性(0-10)：当前消息与上次机器人回复的关联程度
   - 查看上下文中最后一个[我的回复]，判断当前消息是否与之相关
   - 如果当前消息是对上次回复的回应或延续，应给高分
   - 如果当前消息与上次回复完全无关，给中等分数
   - 如果没有上次回复记录，给默认分数5分

回复阈值: {self.reply_threshold} (综合评分达到此分数才回复)

重要！！！请严格按照以下JSON格式回复，不要添加任何其他内容：

请以JSON格式回复：
{{
    "relevance": 分数,
    "willingness": 分数,
    "social": 分数,
    "timing": 分数,
    "continuity": 分数{reasoning_part}
}}

注意：你的回复必须是完整的JSON对象，不要包含任何解释性文字或其他内容！
"""

        try:
            # 使用 provider 调用模型，传入最近的对话历史作为上下文
            recent_contexts = await self._get_recent_contexts(event)

            # 构建完整的判断提示词，将系统提示直接整合到prompt中
            complete_judge_prompt = "你是一个专业的群聊回复决策系统，能够准确判断消息价值和回复时机。"
            if persona_system_prompt:
                complete_judge_prompt += f"\n\n你正在为以下角色的机器人做决策：\n{persona_system_prompt}"
            complete_judge_prompt += "\n\n重要提醒：你必须严格按照JSON格式返回结果，不要包含任何其他内容！请不要进行对话，只返回JSON！\n\n"
            complete_judge_prompt += judge_prompt

            # 重试机制：使用配置的重试次数
            max_retries = self.judge_max_retries + 1  # 配置的次数+原始尝试=总尝试次数
            
            # 如果配置的重试次数为0，只尝试一次
            if self.judge_max_retries == 0:
                max_retries = 1
            
            for attempt in range(max_retries):
                try:
                    logger.debug(f"小参数模型判断尝试 {attempt + 1}/{max_retries}")
                    
                    llm_response = await judge_provider.text_chat(
                        prompt=complete_judge_prompt,
                        contexts=recent_contexts  # 传入最近的对话历史
                    )

                    content = llm_response.completion_text.strip()
                    logger.debug(f"小参数模型原始返回内容: {content[:200]}...")

                    # 尝试提取JSON
                    if content.startswith("```json"):
                        content = content.replace("```json", "").replace("```", "").strip()
                    elif content.startswith("```"):
                        content = content.replace("```", "").strip()

                    judge_data = json.loads(content)

                    # 直接从JSON根对象获取分数
                    relevance = judge_data.get("relevance", 0)
                    willingness = judge_data.get("willingness", 0)
                    social = judge_data.get("social", 0)
                    timing = judge_data.get("timing", 0)
                    continuity = judge_data.get("continuity", 0)
                    
                    # 计算综合评分
                    overall_score = (
                        relevance * self.weights["relevance"] +
                        willingness * self.weights["willingness"] +
                        social * self.weights["social"] +
                        timing * self.weights["timing"] +
                        continuity * self.weights["continuity"]
                    ) / 10.0

                    # 根据综合评分判断是否应该回复
                    should_reply = overall_score >= self.reply_threshold

                    logger.debug(f"小参数模型判断成功，综合评分: {overall_score:.3f}, 是否回复: {should_reply}")

                    return JudgeResult(
                        relevance=relevance,
                        willingness=willingness,
                        social=social,
                        timing=timing,
                        continuity=continuity,
                        reasoning=judge_data.get("reasoning", "") if self.judge_include_reasoning else "",
                        should_reply=should_reply,
                        confidence=overall_score,  # 使用综合评分作为置信度
                        overall_score=overall_score,
                        related_messages=[]  # 不再使用关联消息功能
                    )
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"小参数模型返回JSON解析失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                    logger.warning(f"无法解析的内容: {content[:500]}...")
                    
                    if attempt == max_retries - 1:
                        # 最后一次尝试失败，返回失败结果
                        logger.error(f"小参数模型重试{self.judge_max_retries}次后仍然返回无效JSON，放弃处理")
                        return JudgeResult(should_reply=False, reasoning=f"JSON解析失败，重试{self.judge_max_retries}次")
                    else:
                        # 还有重试机会，添加更强的提示
                        complete_judge_prompt = complete_judge_prompt.replace(
                            "重要提醒：你必须严格按照JSON格式返回结果，不要包含任何其他内容！请不要进行对话，只返回JSON！",
                            f"重要提醒：你必须严格按照JSON格式返回结果，不要包含任何其他内容！请不要进行对话，只返回JSON！这是第{attempt + 2}次尝试，请确保返回有效的JSON格式！"
                        )
                        continue

        except Exception as e:
            logger.error(f"小参数模型判断异常: {e}")
            return JudgeResult(should_reply=False, reasoning=f"异常: {str(e)}")

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=1000)
    async def on_group_message(self, event: AstrMessageEvent):
        """群聊消息处理入口
        
        处理流程：
        1. 同步机器人回复（确保消息顺序正确）
        2. 记录用户消息到缓冲区（所有消息都记录）
        3. 检查是否需要心流判断（@和指令消息跳过）
        4. 小模型判断是否回复
        5. 如需回复，设置唤醒标志让AstrBot核心处理
        """
        
        # === 步骤1：同步机器人回复 ===
        # 在记录用户消息之前，先同步最新的机器人回复
        # 这样可以确保消息的时间顺序正确
        if self.config.get("enable_heartflow", False):
            await self._sync_assistant_messages(event.unified_msg_origin)

        # === 步骤2：记录用户消息 ===
        # 记录所有用户消息到缓冲区，包括@和指令触发的消息
        # 这样即使不进行判断，消息也会被记录下来，保证历史完整
        if (event.get_sender_id() != event.get_self_id() and 
            event.message_str and event.message_str.strip() and
            self.config.get("enable_heartflow", False)):
            
            user_id = event.get_sender_id()
            user_name = event.get_sender_name()
            # 使用与AstrBot相同的格式保存用户信息
            message_content = f"\n[User ID: {user_id}, Nickname: {user_name}]\n{event.message_str}"
            self._record_message(event.unified_msg_origin, "user", message_content)
            logger.debug(f"📝 用户消息已记录到缓冲区 | 群聊: {event.unified_msg_origin[:20]}... | 内容: {event.message_str[:30]}...")

        # === 步骤3：检查是否需要心流判断 ===
        # @和指令触发的消息不需要判断，让AstrBot核心处理
        if not self._should_process_message(event):
            return

        try:
            # === 步骤4：小模型判断 ===
            judge_result = await self.judge_with_tiny_model(event)

            # === 步骤5：根据判断结果处理 ===
            if judge_result.should_reply:
                logger.info(f"🔥 心流触发主动回复 | {event.unified_msg_origin[:20]}... | 评分:{judge_result.overall_score:.2f}")

                # 设置唤醒标志，让AstrBot核心系统处理这条消息
                event.is_at_or_wake_command = True
                
                # 更新主动回复状态（精力消耗、统计等）
                self._update_active_state(event, judge_result)
                logger.info(f"💖 心流设置唤醒标志 | {event.unified_msg_origin[:20]}... | 评分:{judge_result.overall_score:.2f} | {judge_result.reasoning[:50]}...")
                
                # 注意：机器人的回复由AstrBot核心系统生成并保存到conversation_manager
                # 下次用户消息到来时，会在步骤1中自动同步到缓冲区
                
                # 不需要yield任何内容，让核心系统处理
                return
            else:
                # 判断不需要回复，只更新被动状态
                logger.debug(f"心流判断不通过 | {event.unified_msg_origin[:20]}... | 评分:{judge_result.overall_score:.2f} | 原因: {judge_result.reasoning[:30]}...")
                await self._update_passive_state(event, judge_result)

        except Exception as e:
            logger.error(f"心流插件处理消息异常: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _should_process_message(self, event: AstrMessageEvent) -> bool:
        """检查是否应该处理这条消息"""

        # 检查插件是否启用
        if not self.config.get("enable_heartflow", False):
            return False

        # 跳过已经被其他插件或系统标记为唤醒的消息
        if event.is_at_or_wake_command:
            logger.debug(f"跳过已被标记为唤醒的消息: {event.message_str}")
            return False

        # 检查白名单
        if self.whitelist_enabled:
            if not self.chat_whitelist:
                logger.debug(f"白名单为空，跳过处理: {event.unified_msg_origin}")
                return False

            if event.unified_msg_origin not in self.chat_whitelist:
                logger.debug(f"群聊不在白名单中，跳过处理: {event.unified_msg_origin}")
                return False

        # 跳过机器人自己的消息
        if event.get_sender_id() == event.get_self_id():
            return False

        # 跳过空消息
        if not event.message_str or not event.message_str.strip():
            return False

        return True

    def _get_chat_state(self, chat_id: str) -> ChatState:
        """获取群聊状态"""
        if chat_id not in self.chat_states:
            self.chat_states[chat_id] = ChatState()

        # 检查日期重置
        today = datetime.date.today().isoformat()
        state = self.chat_states[chat_id]

        if state.last_reset_date != today:
            state.last_reset_date = today
            # 每日重置时恢复一些精力
            state.energy = min(1.0, state.energy + 0.2)

        return state

    def _get_minutes_since_last_reply(self, chat_id: str) -> int:
        """获取距离上次回复的分钟数"""
        chat_state = self._get_chat_state(chat_id)

        if chat_state.last_reply_time == 0:
            return 999  # 从未回复过

        return int((time.time() - chat_state.last_reply_time) / 60)

    def _record_message(self, chat_id: str, role: str, content: str):
        """记录消息到缓冲区
        
        Args:
            chat_id: 群聊ID
            role: 消息角色（user或assistant）
            content: 消息内容
            
        功能：
            - 将消息添加到指定群聊的缓冲区
            - 自动限制缓冲区大小，防止内存无限增长
        """
        if chat_id not in self.message_buffer:
            self.message_buffer[chat_id] = []
        
        self.message_buffer[chat_id].append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        
        # 限制缓冲区大小，只保留最近的消息
        if len(self.message_buffer[chat_id]) > self.max_buffer_size:
            self.message_buffer[chat_id] = self.message_buffer[chat_id][-self.max_buffer_size:]
    
    async def _sync_assistant_messages(self, chat_id: str):
        """从conversation_manager同步机器人的回复消息到缓冲区
        
        功能：
            - 从AstrBot的conversation_manager获取最新的机器人回复
            - 检查该回复是否已在缓冲区中
            - 如果是新回复，添加到缓冲区
            
        策略：
            - 缓冲区为空时不同步旧消息，避免顺序错乱
            - 只在缓冲区已有消息时同步新回复
            - 采用"从现在开始记录"的策略
        """
        try:
            # 如果缓冲区为空，说明是首次运行或刚重载插件
            # 不同步旧消息，让缓冲区从"现在"开始记录，避免顺序问题
            if chat_id not in self.message_buffer or not self.message_buffer[chat_id]:
                logger.debug(f"缓冲区为空，跳过同步旧消息，从现在开始记录")
                return
            
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(chat_id)
            if not curr_cid:
                return

            conversation = await self.context.conversation_manager.get_conversation(chat_id, curr_cid)
            if not conversation or not conversation.history:
                return

            context = json.loads(conversation.history)
            
            # 从后往前找最后一条assistant消息
            last_assistant_msg = None
            for msg in reversed(context):
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if role == "assistant" and content:
                    last_assistant_msg = content
                    break
            
            if not last_assistant_msg:
                return
            
            # 检查缓冲区最后一条assistant消息
            buffer_msgs = self.message_buffer[chat_id]
            buffer_assistant_msgs = [m for m in buffer_msgs if m.get("role") == "assistant"]
            
            if not buffer_assistant_msgs or buffer_assistant_msgs[-1].get("content") != last_assistant_msg:
                # 新的回复，添加到缓冲区
                self._record_message(chat_id, "assistant", last_assistant_msg)
                logger.debug(f"同步机器人回复到缓冲区: {last_assistant_msg[:30]}...")
                        
        except Exception as e:
            logger.debug(f"同步机器人回复失败: {e}")
    
    async def _get_recent_contexts(self, event: AstrMessageEvent) -> list:
        """获取最近的对话上下文（用于传递给小参数模型）
        
        工作流程：
            1. 使用插件自己的消息缓冲区（包含完整历史，包括未回复的消息）
            2. 为消息添加[群友消息]和[我的回复]标注，帮助小模型识别对话对象
            3. 返回最近N条消息（由context_messages_count配置）
            
        返回格式：
            [
                {"role": "user", "content": "[群友消息] ..."},
                {"role": "assistant", "content": "[我的回复] ..."}
            ]
            
        策略说明：
            - 只使用缓冲区，不回退到conversation_manager
            - 如果缓冲区为空，返回空列表（首次运行时的正常情况）
            - 随着消息积累，缓冲区会逐渐填充完整
            
        注意：
            - 机器人回复的同步已经在on_group_message开始时完成
        """
        chat_id = event.unified_msg_origin
        
        # 使用插件缓冲区
        if chat_id not in self.message_buffer or not self.message_buffer[chat_id]:
            logger.debug(f"缓冲区为空，返回空上下文（首次运行或刚重载）")
            return []
        
        buffer_messages = self.message_buffer[chat_id]
        # 获取最近的 context_messages_count 条消息
        recent_messages = buffer_messages[-self.context_messages_count:] if len(buffer_messages) > self.context_messages_count else buffer_messages
        
        filtered_context = []
        for msg in recent_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role in ["user", "assistant"] and content:
                # 添加标注，帮助小模型识别对话对象
                if role == "user":
                    clean_msg = {
                        "role": role,
                        "content": f"[群友消息] {content}"
                    }
                else:  # assistant
                    clean_msg = {
                        "role": role,
                        "content": f"[我的回复] {content}"
                    }
                filtered_context.append(clean_msg)
        
        logger.info(f"📚 从缓冲区获取到 {len(filtered_context)} 条消息 | 缓冲区总数: {len(buffer_messages)}")
        return filtered_context

    async def _build_chat_context(self, event: AstrMessageEvent) -> str:
        """构建群聊上下文"""
        chat_state = self._get_chat_state(event.unified_msg_origin)

        context_info = f"""最近活跃度: {'高' if chat_state.total_messages > 100 else '中' if chat_state.total_messages > 20 else '低'}
历史回复率: {(chat_state.total_replies / max(1, chat_state.total_messages) * 100):.1f}%
当前时间: {datetime.datetime.now().strftime('%H:%M')}"""
        return context_info


    def _update_active_state(self, event: AstrMessageEvent, judge_result: JudgeResult):
        """更新主动回复状态"""
        chat_id = event.unified_msg_origin
        chat_state = self._get_chat_state(chat_id)

        # 更新回复相关状态
        chat_state.last_reply_time = time.time()
        chat_state.total_replies += 1
        chat_state.total_messages += 1

        # 精力消耗（回复后精力下降）
        chat_state.energy = max(0.1, chat_state.energy - self.energy_decay_rate)

        logger.debug(f"更新主动状态: {chat_id[:20]}... | 精力: {chat_state.energy:.2f}")

    async def _update_passive_state(self, event: AstrMessageEvent, judge_result: JudgeResult):
        """更新被动状态（未回复）
        
        功能：
            - 更新消息统计
            - 恢复精力值（不回复时精力缓慢恢复）
            - 记录判断日志
            
        注意：
            - 用户消息已经在on_group_message开始时记录到缓冲区了
            - 这里只需要更新状态，不需要再记录消息
        """
        chat_id = event.unified_msg_origin
        chat_state = self._get_chat_state(chat_id)

        # 更新消息计数
        chat_state.total_messages += 1

        # 精力恢复（不回复时精力缓慢恢复）
        chat_state.energy = min(1.0, chat_state.energy + self.energy_recovery_rate)

        logger.debug(f"更新被动状态: {chat_id[:20]}... | 精力: {chat_state.energy:.2f} | 原因: {judge_result.reasoning[:30]}...")

    # 管理员命令：查看心流状态
    @filter.command("heartflow")
    async def heartflow_status(self, event: AstrMessageEvent):
        """查看心流状态"""

        chat_id = event.unified_msg_origin
        chat_state = self._get_chat_state(chat_id)

        status_info = f"""
心流状态报告

当前状态:
- 群聊ID: {event.unified_msg_origin}
- 精力水平: {chat_state.energy:.2f}/1.0 {'高' if chat_state.energy > 0.7 else '中' if chat_state.energy > 0.3 else '低'}
- 上次回复: {self._get_minutes_since_last_reply(chat_id)}分钟前

历史统计:
- 总消息数: {chat_state.total_messages}
- 总回复数: {chat_state.total_replies}
- 回复率: {(chat_state.total_replies / max(1, chat_state.total_messages) * 100):.1f}%

配置参数:
- 回复阈值: {self.reply_threshold}
- 判断提供商: {self.judge_provider_name}
- 最大重试次数: {self.judge_max_retries}
- 白名单模式: {'开启' if self.whitelist_enabled else '关闭'}
- 白名单群聊数: {len(self.chat_whitelist) if self.whitelist_enabled else 0}

智能缓存:
- 系统提示词缓存: {len(self.system_prompt_cache)} 个
- 消息缓冲区: {len(self.message_buffer[chat_id]) if chat_id in self.message_buffer else 0}/{self.max_buffer_size} 条

评分权重:
- 内容相关度: {self.weights['relevance']:.0%}
- 回复意愿: {self.weights['willingness']:.0%}
- 社交适宜性: {self.weights['social']:.0%}
- 时机恰当性: {self.weights['timing']:.0%}
- 对话连贯性: {self.weights['continuity']:.0%}

插件状态: {'已启用' if self.config.get('enable_heartflow', False) else '已禁用'}
"""

        event.set_result(event.plain_result(status_info))

    # 管理员命令：重置心流状态
    @filter.command("heartflow_reset")
    async def heartflow_reset(self, event: AstrMessageEvent):
        """重置心流状态"""

        chat_id = event.unified_msg_origin
        if chat_id in self.chat_states:
            del self.chat_states[chat_id]

        event.set_result(event.plain_result("心流状态已重置"))
        logger.info(f"心流状态已重置: {chat_id}")

    # 管理员命令：查看系统提示词缓存
    @filter.command("heartflow_cache")
    async def heartflow_cache_status(self, event: AstrMessageEvent):
        """查看系统提示词缓存状态"""
        
        cache_info = "系统提示词缓存状态\n\n"
        
        if not self.system_prompt_cache:
            cache_info += "当前无缓存记录"
        else:
            cache_info += f"总缓存数量: {len(self.system_prompt_cache)}\n\n"
            
            for cache_key, cache_data in self.system_prompt_cache.items():
                original_len = len(cache_data.get("original", ""))
                summarized_len = len(cache_data.get("summarized", ""))
                persona_id = cache_data.get("persona_id", "unknown")
                
                cache_info += f"缓存键: {cache_key}\n"
                cache_info += f"人格ID: {persona_id}\n"
                cache_info += f"压缩率: {original_len} -> {summarized_len} ({(1-summarized_len/max(1,original_len))*100:.1f}% 压缩)\n"
                cache_info += f"精简内容: {cache_data.get('summarized', '')[:100]}...\n\n"
        
        event.set_result(event.plain_result(cache_info))

    # 管理员命令：清除系统提示词缓存
    @filter.command("heartflow_cache_clear")
    async def heartflow_cache_clear(self, event: AstrMessageEvent):
        """清除系统提示词缓存"""
        
        cache_count = len(self.system_prompt_cache)
        self.system_prompt_cache.clear()
        
        event.set_result(event.plain_result(f"已清除 {cache_count} 个系统提示词缓存"))
        logger.info(f"系统提示词缓存已清除，共清除 {cache_count} 个缓存")
    
    # 管理员命令：查看消息缓冲区状态
    @filter.command("heartflow_buffer")
    async def heartflow_buffer_status(self, event: AstrMessageEvent):
        """查看消息缓冲区状态"""
        
        chat_id = event.unified_msg_origin
        
        buffer_info = "消息缓冲区状态\n\n"
        
        if chat_id not in self.message_buffer or not self.message_buffer[chat_id]:
            buffer_info += "当前群聊缓冲区为空"
        else:
            buffer = self.message_buffer[chat_id]
            buffer_info += f"缓冲区消息数量: {len(buffer)}/{self.max_buffer_size}\n\n"
            buffer_info += "最近10条消息:\n"
            
            recent_10 = buffer[-10:] if len(buffer) > 10 else buffer
            for i, msg in enumerate(recent_10, 1):
                role = msg.get("role", "")
                content = msg.get("content", "")
                role_text = "群友" if role == "user" else "我"
                buffer_info += f"{i}. [{role_text}] {content[:50]}...\n"
        
        event.set_result(event.plain_result(buffer_info))
    
    # 管理员命令：清除消息缓冲区
    @filter.command("heartflow_buffer_clear")
    async def heartflow_buffer_clear(self, event: AstrMessageEvent):
        """清除当前群聊的消息缓冲区"""
        
        chat_id = event.unified_msg_origin
        if chat_id in self.message_buffer:
            msg_count = len(self.message_buffer[chat_id])
            del self.message_buffer[chat_id]
            event.set_result(event.plain_result(f"已清除当前群聊的消息缓冲区（{msg_count} 条消息）"))
            logger.info(f"消息缓冲区已清除: {chat_id} ({msg_count} 条)")
        else:
            event.set_result(event.plain_result("当前群聊缓冲区为空，无需清除"))

    async def _get_persona_system_prompt(self, event: AstrMessageEvent) -> str:
        """获取当前对话的人格系统提示词"""
        try:
            # 获取当前对话
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(event.unified_msg_origin)
            if not curr_cid:
                # 如果没有对话ID，使用默认人格
                default_persona_name = self.context.provider_manager.selected_default_persona["name"]
                return self._get_persona_prompt_by_name(default_persona_name)

            conversation = await self.context.conversation_manager.get_conversation(event.unified_msg_origin, curr_cid)
            if not conversation:
                # 如果没有对话对象，使用默认人格
                default_persona_name = self.context.provider_manager.selected_default_persona["name"]
                return self._get_persona_prompt_by_name(default_persona_name)

            # 获取人格ID
            persona_id = conversation.persona_id

            if not persona_id:
                # persona_id 为 None 时，使用默认人格
                persona_id = self.context.provider_manager.selected_default_persona["name"]
            elif persona_id == "[%None]":
                # 用户显式取消人格时，不使用任何人格
                return ""

            return self._get_persona_prompt_by_name(persona_id)

        except Exception as e:
            logger.debug(f"获取人格系统提示词失败: {e}")
            return ""

    def _get_persona_prompt_by_name(self, persona_name: str) -> str:
        """根据人格名称获取人格提示词"""
        try:
            # 从provider_manager中查找人格
            for persona in self.context.provider_manager.personas:
                if persona["name"] == persona_name:
                    return persona.get("prompt", "")

            logger.debug(f"未找到人格: {persona_name}")
            return ""

        except Exception as e:
            logger.debug(f"获取人格提示词失败: {e}")
            return ""
