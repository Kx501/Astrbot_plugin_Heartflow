import json
import time
import datetime
import asyncio
from typing import Dict
from dataclasses import dataclass
from pathlib import Path

import astrbot.api.star as star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api import logger
from astrbot.api.star import StarTools
from astrbot.api.provider import LLMResponse


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
    # 好感度系统
    user_favorability: Dict[str, float] = None  # {user_id: favorability (0-100)}
    user_interaction_count: Dict[str, int] = None  # {user_id: count}
    last_favorability_decay: str = ""  # 上次好感度衰减日期
    
    def __post_init__(self):
        if self.user_favorability is None:
            self.user_favorability = {}
        if self.user_interaction_count is None:
            self.user_interaction_count = {}



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
        # 1. 用户消息：在on_group_message中实时记录
        # 2. 机器人回复：在on_llm_response钩子中实时记录
        # 3. 判断时：使用缓冲区的完整历史
        #
        # 注意：缓冲区采用"从现在开始记录"策略，不回溯历史
        self.message_buffer: Dict[str, list] = {}
        self.max_buffer_size = self.config.get("max_buffer_size", 50)  # 每个群聊最多缓存50条
        
        # 判断状态标记：用于过滤小模型的判断结果
        self.judging_sessions: set = set()  # 正在进行判断的会话ID集合

        # 判断配置
        self.judge_include_reasoning = self.config.get("judge_include_reasoning", True)
        self.judge_max_retries = max(0, self.config.get("judge_max_retries", 3))  # 确保最小为0
        
        # 提示词配置
        self.judge_evaluation_rules = self.config.get("judge_evaluation_rules", "")
        self.summarize_instruction = self.config.get("summarize_instruction", "")
        
        # 好感度系统配置
        self.enable_favorability = self.config.get("enable_favorability", False)
        self.enable_global_favorability = self.config.get("enable_global_favorability", False)
        self.favorability_impact_strength = self.config.get("favorability_impact_strength", 1.0)
        self.favorability_decay_daily = self.config.get("favorability_decay_daily", 1.0)
        self.initial_favorability = self.config.get("initial_favorability", 10.0)  # 新用户初始好感度
        
        # 全局好感度存储：{user_id: favorability}
        # 跨群聊的用户好感度，不受白名单限制
        self.global_favorability: Dict[str, float] = {}
        self.global_interaction_count: Dict[str, int] = {}
        
        # 好感度计算权重
        self.fav_weights = {
            "relevance": self.config.get("fav_weight_relevance", 0.4),
            "social": self.config.get("fav_weight_social", 0.3),
            "continuity": self.config.get("fav_weight_continuity", 0.2),
            "willingness": self.config.get("fav_weight_willingness", 0.05),
            "timing": self.config.get("fav_weight_timing", 0.05)
        }
        
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

        # 获取插件数据目录
        try:
            self.data_dir = StarTools.get_data_dir(None)  # 自动检测插件名称
            self.favorability_file = self.data_dir / "favorability.json"
            self.global_favorability_file = self.data_dir / "global_favorability.json"
            logger.info(f"插件数据目录: {self.data_dir}")
        except Exception as e:
            logger.error(f"获取数据目录失败，好感度系统已禁用: {e}")
            self.enable_favorability = False  # 获取路径失败，关闭好感度系统
            self.data_dir = None
            self.favorability_file = None
            self.global_favorability_file = None
        
        # 加载好感度数据
        if self.enable_favorability:
            self._load_favorability()
            # 启动自动保存任务
            asyncio.create_task(self._auto_save_task())

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
            
            # 使用配置的总结指令，如果没有配置则使用默认指令
            if self.summarize_instruction:
                instruction = self.summarize_instruction
            else:
                instruction = "请将以下机器人角色设定总结为简洁的核心要点，保留关键的性格特征、行为方式和角色定位。总结后的内容应该在100-200字以内，突出最重要的角色特点。"
            
            summarize_prompt = f"""{instruction}

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
        
        session_id = event.unified_msg_origin
        
        # 标记开始判断（用于过滤小模型回复）
        self.judging_sessions.add(session_id)
        
        try:
            return await self._do_judge(event)
        finally:
            # 判断结束，移除标记
            self.judging_sessions.discard(session_id)
    
    async def _do_judge(self, event: AstrMessageEvent) -> JudgeResult:
        """执行判断的内部方法"""

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

        reasoning_part = ""
        if self.judge_include_reasoning:
            reasoning_part = ',\n    "reasoning": "详细分析原因，说明为什么应该或不应该回复，需要结合机器人角色特点进行分析，特别说明与上次回复的关联性"'

        # 使用配置的评估规则，如果没有配置则使用默认规则
        if self.judge_evaluation_rules:
            evaluation_rules = self.judge_evaluation_rules
        else:
            evaluation_rules = """请从以下5个维度评估（0-10分），重要提醒：基于上述机器人角色设定来判断是否适合回复：

1. 内容相关度(0-10)：消息是否有趣、有价值、适合我回复
   - 考虑消息的质量、话题性、是否需要回应
   - 识别并过滤垃圾消息、无意义内容
   - 结合机器人角色特点，判断是否符合角色定位

2. 回复意愿(0-10)：基于当前状态，我回复此消息的意愿
   - 考虑当前精力水平和对用户的印象
   - 考虑今日回复频率控制
   - 基于机器人角色设定，判断是否应该主动参与此话题

3. 社交适宜性(0-10)：在当前群聊氛围下回复是否合适
   - 考虑群聊活跃度和讨论氛围
   - 考虑机器人角色在群中的定位和表现方式

4. 时机恰当性(0-10)：回复时机是否恰当
   - 考虑距离上次回复的时间间隔
   - 考虑消息的紧急性和时效性

5. 对话连贯性(0-10)：当前消息与上次机器人回复的关联程度
   - 查看对话历史中最后的[我的回复]
   - 如果当前消息是对我上次回复的回应或延续，给高分
   - 如果当前消息与我上次回复无关，给中等分数
   - 如果对话历史中没有我的回复记录，给低分"""

        # 获取好感度信息
        user_id = event.get_sender_id()
        user_fav = self._get_user_favorability(event.unified_msg_origin, user_id)
        interaction_count = self._get_user_interaction_count(event.unified_msg_origin, user_id)
        
        # 好感度描述
        fav_info = ""
        if self.enable_favorability:
            level, emoji = self._get_favorability_level(user_fav)
            fav_info = f"\n对当前用户的好感度: {user_fav:.0f}/100 ({level} {emoji})\n互动历史: {interaction_count}次"
        
        # 构建完整的判断提示词
        judge_prompt = f"""
你是群聊机器人的决策系统，需要判断是否应该主动回复以下消息。

重要说明：
- 对话历史已提供给你，你可以查看完整的对话流程
- [群友消息] = 群友发送的消息
- [我的回复] = 机器人（我）发送的回复

机器人角色设定:
{persona_system_prompt if persona_system_prompt else "默认角色：智能助手"}

当前群聊ID:
{event.unified_msg_origin}

机器人状态:
我的精力水平: {chat_state.energy:.1f}/1.0
最近活跃度: {'高' if chat_state.total_messages > 100 else '中' if chat_state.total_messages > 20 else '低'}
上次发言: {self._get_minutes_since_last_reply(event.unified_msg_origin)}分钟前
历史回复率: {(chat_state.total_replies / max(1, chat_state.total_messages) * 100):.1f}%{fav_info}

待判断消息:
发送者: {event.get_sender_name()}
内容: {event.message_str}
时间: {datetime.datetime.now().strftime('%H:%M:%S')}

{evaluation_rules}

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

                    # 应用好感度调整
                    threshold_adjustment = self._get_threshold_adjustment(user_fav)
                    adjusted_threshold = self.reply_threshold + threshold_adjustment
                    
                    # 根据调整后的阈值判断是否应该回复
                    should_reply = overall_score >= adjusted_threshold
                    
                    if self.enable_favorability and abs(threshold_adjustment) > 0.01:
                        logger.debug(f"好感度调整阈值: {self.reply_threshold:.2f} → {adjusted_threshold:.2f} (好感度:{user_fav:.0f})")

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
        1. 记录用户消息到缓冲区（所有消息都记录）
        2. 检查是否需要心流判断（@和指令消息跳过）
        3. 小模型判断是否回复
        4. 如需回复，设置唤醒标志让AstrBot核心处理
        
        注意：机器人回复通过on_llm_response钩子实时记录
        """
        
        # === 步骤1：记录用户消息 ===
        # 注意：机器人回复通过on_llm_response钩子实时记录，不需要同步
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
                
                # 更新好感度（回复了）
                if self.enable_favorability:
                    user_id = event.get_sender_id()
                    fav_delta = self._calculate_favorability_change(judge_result, did_reply=True)
                    self._update_favorability(event.unified_msg_origin, user_id, fav_delta)
                    self._record_interaction(event.unified_msg_origin, user_id)
                
                logger.info(f"💖 心流设置唤醒标志 | {event.unified_msg_origin[:20]}... | 评分:{judge_result.overall_score:.2f} | {judge_result.reasoning[:50]}...")
                
                # 注意：机器人的回复由AstrBot核心系统生成并保存到conversation_manager
                # 下次用户消息到来时，会在步骤1中自动同步到缓冲区
                
                # 不需要yield任何内容，让核心系统处理
                return
            else:
                # 判断不需要回复，只更新被动状态
                logger.debug(f"心流判断不通过 | {event.unified_msg_origin[:20]}... | 评分:{judge_result.overall_score:.2f} | 原因: {judge_result.reasoning[:30]}...")
                await self._update_passive_state(event, judge_result)
                
                # 更新好感度（没回复）
                if self.enable_favorability:
                    user_id = event.get_sender_id()
                    fav_delta = self._calculate_favorability_change(judge_result, did_reply=False)
                    self._update_favorability(event.unified_msg_origin, user_id, fav_delta)
                    self._record_interaction(event.unified_msg_origin, user_id)

        except Exception as e:
            logger.error(f"心流插件处理消息异常: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    @filter.on_llm_response()
    async def on_llm_resp(self, event: AstrMessageEvent, resp: LLMResponse):
        """LLM回复完成时，立即记录机器人回复到缓冲区
        
        优势：
        - 实时记录，不需要同步
        - 支持连续回复（每次回复都触发）
        - 顺序完美，不会错乱
        
        过滤机制：
        - 跳过小模型判断结果（通过judging_sessions标记）
        - 只记录群聊消息
        - 白名单检查（如果启用）
        """
        if not self.config.get("enable_heartflow", False):
            return
        
        try:
            chat_id = event.unified_msg_origin
            
            # === 检查1：跳过小模型判断 ===
            if chat_id in self.judging_sessions:
                logger.debug("跳过小模型判断结果")
                return
            
            # === 检查2：只记录群聊消息 ===
            # 避免记录私聊或其他类型的消息
            if event.message_obj.type.name != "GROUP_MESSAGE":
                return
            
            # === 检查3：白名单检查 ===
            if self.whitelist_enabled:
                if not self.chat_whitelist or chat_id not in self.chat_whitelist:
                    return
            
            # === 记录机器人回复 ===
            assistant_reply = resp.completion_text
            
            if assistant_reply and assistant_reply.strip():
                self._record_message(chat_id, "assistant", assistant_reply)
                logger.debug(f"📝 机器人回复已记录到缓冲区: {assistant_reply[:30]}...")
        
        except Exception as e:
            logger.debug(f"记录机器人回复失败: {e}")

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
        
        # 好感度每日衰减
        if self.enable_favorability and state.last_favorability_decay != today:
            state.last_favorability_decay = today
            self._apply_favorability_decay(chat_id)

        return state

    def _get_minutes_since_last_reply(self, chat_id: str) -> int:
        """获取距离上次回复的分钟数"""
        chat_state = self._get_chat_state(chat_id)

        if chat_state.last_reply_time == 0:
            return 999  # 从未回复过

        return int((time.time() - chat_state.last_reply_time) / 60)

    # ===== 好感度系统方法 =====
    
    def _load_favorability(self):
        """从文件加载好感度数据"""
        try:
            # 加载群聊好感度
            if self.favorability_file.exists():
                with open(self.favorability_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 恢复数据到chat_states
                for chat_id, chat_data in data.items():
                    if chat_id not in self.chat_states:
                        self.chat_states[chat_id] = ChatState()
                    
                    state = self.chat_states[chat_id]
                    state.user_favorability = chat_data.get("favorability", {})
                    state.user_interaction_count = chat_data.get("interaction_count", {})
                    state.last_favorability_decay = chat_data.get("last_decay", "")
                
                logger.info(f"群聊好感度数据已加载，共{len(data)}个群聊")
            else:
                logger.info("未找到群聊好感度数据文件，使用默认值")
            
            # 加载全局好感度
            if self.enable_global_favorability and self.global_favorability_file.exists():
                with open(self.global_favorability_file, 'r', encoding='utf-8') as f:
                    global_data = json.load(f)
                
                self.global_favorability = global_data.get("favorability", {})
                self.global_interaction_count = global_data.get("interaction_count", {})
                
                logger.info(f"全局好感度数据已加载，共{len(self.global_favorability)}个用户")

        except Exception as e:
            logger.error(f"加载好感度数据失败: {e}")
    
    def _save_favorability(self):
        """保存好感度数据到文件"""
        try:
            # 保存群聊好感度
            data = {}
            for chat_id, state in self.chat_states.items():
                if state.user_favorability:  # 只保存有数据的群聊
                    data[chat_id] = {
                        "favorability": state.user_favorability,
                        "interaction_count": state.user_interaction_count,
                        "last_decay": state.last_favorability_decay
                    }
            
            # 确保目录存在
            self.favorability_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存群聊好感度
            with open(self.favorability_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"群聊好感度数据已保存，共{len(data)}个群聊")
            
            # 保存全局好感度
            if self.enable_global_favorability:
                global_data = {
                    "favorability": self.global_favorability,
                    "interaction_count": self.global_interaction_count
                }
                
                with open(self.global_favorability_file, 'w', encoding='utf-8') as f:
                    json.dump(global_data, f, ensure_ascii=False, indent=2)
                
                logger.debug(f"全局好感度数据已保存，共{len(self.global_favorability)}个用户")

        except Exception as e:
            logger.error(f"保存好感度数据失败: {e}")
    
    async def _auto_save_task(self):
        """定期自动保存好感度数据"""
        try:
            while True:
                await asyncio.sleep(300)  # 每5分钟保存一次
                if self.enable_favorability:
                    self._save_favorability()
                    logger.debug("好感度数据已自动保存")
        except asyncio.CancelledError:
            # 任务被取消，保存数据
            self._save_favorability()
            logger.info("自动保存任务已停止")
        except Exception as e:
            logger.error(f"自动保存任务异常: {e}")
    
    def _get_user_favorability(self, chat_id: str, user_id: str) -> float:
        """获取用户好感度（0-100）
        
        优先级：
        1. 如果启用全局好感度且满足条件，返回全局好感度
        2. 否则返回群聊本地好感度
        
        全局好感度条件：
        - enable_global_favorability = True
        - 如果启用了白名单，当前群聊必须在白名单中
        
        新用户默认好感度：由initial_favorability配置（默认30）
        """
        if not self.enable_favorability:
            return self.initial_favorability  # 系统未启用，返回初始值
        
        # 检查是否使用全局好感度
        use_global = self.enable_global_favorability
        
        # 如果启用了白名单，全局好感度也受白名单控制
        if use_global and self.whitelist_enabled:
            if not self.chat_whitelist or chat_id not in self.chat_whitelist:
                use_global = False  # 不在白名单中，不使用全局好感度
        
        if use_global and user_id in self.global_favorability:
            return self.global_favorability[user_id]
        
        # 使用群聊本地好感度
        chat_state = self._get_chat_state(chat_id)
        return chat_state.user_favorability.get(user_id, self.initial_favorability)
    
    def _get_user_interaction_count(self, chat_id: str, user_id: str) -> int:
        """获取用户互动次数"""
        chat_state = self._get_chat_state(chat_id)
        return chat_state.user_interaction_count.get(user_id, 0)
    
    def _get_favorability_level(self, favorability: float) -> tuple:
        """获取好感度等级和emoji
        
        返回: (等级名称, emoji)
        """
        if favorability >= 80:
            return ("挚友", "💖")
        elif favorability >= 65:
            return ("好友", "😊")
        elif favorability >= 50:
            return ("熟人", "🙂")
        elif favorability >= 35:
            return ("普通", "😐")
        elif favorability >= 20:
            return ("陌生", "😑")
        else:
            return ("冷淡", "😒")
    
    def _calculate_favorability_change(self, judge_result: JudgeResult, did_reply: bool) -> float:
        """基于5个维度的归一化分数计算好感度变化
        
        优点：
        - 不依赖具体分数阈值
        - 适用于不同AI模型的评分习惯
        - 基于相对值而非绝对值
        
        返回：-5.0 到 +5.0 的变化值
        """
        if not self.enable_favorability:
            return 0.0
        
        # === 归一化5个维度（0-10 → 0-1） ===
        norm_relevance = judge_result.relevance / 10.0
        norm_social = judge_result.social / 10.0
        norm_continuity = judge_result.continuity / 10.0
        norm_willingness = judge_result.willingness / 10.0
        norm_timing = judge_result.timing / 10.0
        
        # === 计算综合质量分（0-1） ===
        quality_score = (
            norm_relevance * self.fav_weights["relevance"] +
            norm_social * self.fav_weights["social"] +
            norm_continuity * self.fav_weights["continuity"] +
            norm_willingness * self.fav_weights["willingness"] +
            norm_timing * self.fav_weights["timing"]
        )
        
        # === 映射到好感度变化（-5 到 +3） ===
        # 使用分段线性映射
        if quality_score > 0.8:
            # 非常好的互动 → +2 到 +3
            delta = 2.0 + (quality_score - 0.8) / 0.2 * 1.0
        elif quality_score > 0.6:
            # 良好的互动 → +0.8 到 +2
            delta = 0.8 + (quality_score - 0.6) / 0.2 * 1.2
        elif quality_score > 0.4:
            # 普通互动 → -1.0 到 +0.8
            delta = -1.0 + (quality_score - 0.4) / 0.2 * 1.8
        elif quality_score > 0.2:
            # 较差互动 → -2.5 到 -1.0
            delta = -2.5 + (quality_score - 0.2) / 0.2 * 1.5
        else:
            # 很差的互动 → -5 到 -2.5
            delta = -5.0 + quality_score / 0.2 * 2.5
        
        # === 互动结果修正 ===
        if did_reply:
            # 回复了，说明互动成功，轻微加成
            delta += 0.3
        else:
            # 没回复，如果质量还可以，轻微减少好感
            if quality_score > 0.5:
                delta -= 0.2
        
        # === 限制范围 ===
        return max(-5.0, min(5.0, delta))
    
    def _update_favorability(self, chat_id: str, user_id: str, delta: float):
        """更新用户好感度
        
        根据配置同时更新：
        1. 群聊本地好感度（总是更新）
        2. 全局好感度（如果启用且满足白名单条件）
        """
        if not self.enable_favorability:
            return
        
        # 更新群聊本地好感度
        chat_state = self._get_chat_state(chat_id)
        
        current = chat_state.user_favorability.get(user_id, self.initial_favorability)
        new_value = max(0.0, min(100.0, current + delta))
        chat_state.user_favorability[user_id] = new_value
        
        # 更新全局好感度（如果启用）
        if self.enable_global_favorability:
            # 检查白名单限制
            can_update_global = True
            if self.whitelist_enabled:
                if not self.chat_whitelist or chat_id not in self.chat_whitelist:
                    can_update_global = False  # 不在白名单中，不更新全局好感度
            
            if can_update_global:
                global_current = self.global_favorability.get(user_id, self.initial_favorability)
                global_new = max(0.0, min(100.0, global_current + delta))
                self.global_favorability[user_id] = global_new
        
        if abs(delta) > 0.1:  # 只记录有意义的变化
            logger.debug(f"好感度更新: {user_id[-4:]}... | 本地:{current:.1f}→{new_value:.1f} ({delta:+.1f})")
    
    def _record_interaction(self, chat_id: str, user_id: str):
        """记录用户互动次数
        
        同时更新：
        1. 群聊本地互动计数
        2. 全局互动计数（如果启用且满足白名单条件）
        """
        if not self.enable_favorability:
            return
        
        # 更新群聊本地互动计数
        chat_state = self._get_chat_state(chat_id)
        chat_state.user_interaction_count[user_id] = \
            chat_state.user_interaction_count.get(user_id, 0) + 1
        
        # 更新全局互动计数（如果启用）
        if self.enable_global_favorability:
            # 检查白名单限制
            can_update_global = True
            if self.whitelist_enabled:
                if not self.chat_whitelist or chat_id not in self.chat_whitelist:
                    can_update_global = False
            
            if can_update_global:
                self.global_interaction_count[user_id] = \
                    self.global_interaction_count.get(user_id, 0) + 1
    
    def _apply_favorability_decay(self, chat_id: str):
        """应用好感度自然衰减
        
        设计理念：
        - 好感度会随时间自然向50（中性）回归
        - 高好感度衰减更快（避免永久高好感）
        - 低好感度恢复更快（给用户改过机会）
        """
        chat_state = self._get_chat_state(chat_id)
        decay_rate = self.favorability_decay_daily
        
        for user_id in list(chat_state.user_favorability.keys()):
            current = chat_state.user_favorability[user_id]
            
            if current > 50:
                # 高好感度向中性回归（衰减稍快）
                decay = min(current - 50, decay_rate * 1.5)
                chat_state.user_favorability[user_id] = current - decay
            elif current < 50:
                # 低好感度向中性恢复（恢复更快）
                recovery = min(50 - current, decay_rate * 2.0)
                chat_state.user_favorability[user_id] = current + recovery
    
    def _get_threshold_adjustment(self, favorability: float) -> float:
        """根据好感度计算回复阈值调整
        
        使用平滑曲线，避免阈值突变
        
        Args:
            favorability: 0-100
        
        Returns:
            阈值调整值（-0.2 到 +0.2）
        """
        if not self.enable_favorability:
            return 0.0
        
        # 归一化到 -1 到 +1
        normalized = (favorability - 50) / 50
        
        # 使用线性映射
        # 好感度100 → -0.2（更容易回复）
        # 好感度50 → 0
        # 好感度0 → +0.2（更难触发回复）
        adjustment = -normalized * 0.2 * self.favorability_impact_strength
        
        return adjustment

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
    
    async def _get_recent_contexts(self, event: AstrMessageEvent) -> list:
        """获取最近的对话上下文（用于传递给小参数模型）
        
        工作流程：
            1. 从插件的消息缓冲区获取消息（包含完整历史，包括未回复的消息）
            2. 为消息添加[群友消息]和[我的回复]标注，帮助小模型识别对话对象
            3. 返回最近N条消息（由context_messages_count配置）
            
        返回格式：
            [
                {"role": "user", "content": "[群友消息] ..."},
                {"role": "assistant", "content": "[我的回复] ..."}
            ]
            
        消息来源：
            - 用户消息：在on_group_message中实时记录
            - 机器人回复：在on_llm_response钩子中实时记录
            - 无需同步，消息都是实时添加的
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
        
        # 好感度统计
        fav_stats = ""
        if self.enable_favorability:
            # 根据是否启用全局好感度，选择数据源
            fav_data = self.global_favorability if self.enable_global_favorability else chat_state.user_favorability
            fav_scope = "全局" if self.enable_global_favorability else "当前群聊"
            
            total_users = len(fav_data)
            if total_users > 0:
                avg_fav = sum(fav_data.values()) / total_users
                high_fav = len([f for f in fav_data.values() if f >= 70])
                low_fav = len([f for f in fav_data.values() if f <= 30])
                fav_stats = f"""
好感度统计（{fav_scope}）:
- 记录用户数: {total_users}
- 平均好感度: {avg_fav:.1f}/100
- 高好感用户: {high_fav}个 (≥70)
- 低好感用户: {low_fav}个 (≤30)
"""

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

{fav_stats}
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
    
    # 管理员命令：查看好感度
    @filter.command("heartflow_fav")
    async def heartflow_favorability(self, event: AstrMessageEvent):
        """查看当前用户的好感度"""
        
        if not self.enable_favorability:
            event.set_result(event.plain_result("好感度系统未启用"))
            return
        
        chat_id = event.unified_msg_origin
        user_id = event.get_sender_id()
        user_name = event.get_sender_name()
        
        # 获取实际使用的好感度
        user_fav = self._get_user_favorability(chat_id, user_id)
        interaction_count = self._get_user_interaction_count(chat_id, user_id)
        level, emoji = self._get_favorability_level(user_fav)
        threshold_adj = self._get_threshold_adjustment(user_fav)
        
        # 判断使用的是全局还是群聊好感度
        fav_source = "全局（跨群聊）" if (self.enable_global_favorability and user_id in self.global_favorability) else "当前群聊"
        
        fav_info = f"""
好感度报告

用户：{user_name}
用户ID：{user_id}

好感度：{user_fav:.1f}/100 {emoji}
关系等级：{level}
互动次数：{interaction_count}次
数据范围：{fav_source}

影响效果：
- 回复阈值调整：{threshold_adj:+.3f}
- 实际阈值：{self.reply_threshold + threshold_adj:.3f}（原始：{self.reply_threshold}）
- {'更容易获得回复' if threshold_adj < 0 else '更难获得回复' if threshold_adj > 0 else '无影响'}

系统状态：
- 好感度影响强度：{self.favorability_impact_strength}
- 每日衰减速度：{self.favorability_decay_daily}
"""
        
        event.set_result(event.plain_result(fav_info))
    
    # 管理员命令：好感度排行榜
    @filter.command("heartflow_fav_rank")
    async def heartflow_favorability_rank(self, event: AstrMessageEvent):
        """查看当前群聊的好感度排行榜"""
        
        if not self.enable_favorability:
            event.set_result(event.plain_result("好感度系统未启用"))
            return
        
        chat_id = event.unified_msg_origin
        chat_state = self._get_chat_state(chat_id)
        
        if not chat_state.user_favorability:
            event.set_result(event.plain_result("当前群聊暂无好感度记录"))
            return
        
        # 排序
        sorted_users = sorted(
            chat_state.user_favorability.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        result = "好感度排行榜\n\n"
        for i, (uid, fav) in enumerate(sorted_users[:10], 1):
            level, emoji = self._get_favorability_level(fav)
            interaction = chat_state.user_interaction_count.get(uid, 0)
            result += f"{i}. 用户{uid[-6:]}: {fav:.0f}/100 {emoji} ({level}, {interaction}次互动)\n"
        
        if len(sorted_users) > 10:
            result += f"\n...还有{len(sorted_users) - 10}个用户"
        
        event.set_result(event.plain_result(result))
    
    # 管理员命令：重置好感度
    @filter.command("heartflow_fav_reset")
    async def heartflow_favorability_reset(self, event: AstrMessageEvent):
        """重置当前群聊所有用户的好感度"""
        
        if not self.enable_favorability:
            event.set_result(event.plain_result("好感度系统未启用"))
            return
        
        chat_id = event.unified_msg_origin
        chat_state = self._get_chat_state(chat_id)
        
        user_count = len(chat_state.user_favorability)
        chat_state.user_favorability.clear()
        chat_state.user_interaction_count.clear()
        
        event.set_result(event.plain_result(f"已重置当前群聊所有用户的好感度（{user_count}个用户）"))
        logger.info(f"好感度已重置: {chat_id} ({user_count}个用户)")

    # 管理员命令：手动保存好感度
    @filter.command("heartflow_fav_save")
    async def heartflow_favorability_save(self, event: AstrMessageEvent):
        """手动保存好感度数据"""
        
        if not self.enable_favorability:
            event.set_result(event.plain_result("好感度系统未启用"))
            return
        
        try:
            self._save_favorability()
            
            # 统计保存的数据
            total_chats = 0
            total_users = 0
            for state in self.chat_states.values():
                if state.user_favorability:
                    total_chats += 1
                    total_users += len(state.user_favorability)
            
            event.set_result(event.plain_result(
                f"好感度数据已保存\n\n"
                f"保存位置: {self.favorability_file}\n"
                f"群聊数: {total_chats}\n"
                f"用户数: {total_users}"
            ))
            logger.info(f"手动保存好感度数据: {total_chats}个群聊, {total_users}个用户")
        except Exception as e:
            event.set_result(event.plain_result(f"保存失败: {e}"))
            logger.error(f"手动保存好感度失败: {e}")
    
    async def terminate(self):
        """插件卸载/停用时调用，保存数据"""
        if self.enable_favorability:
            self._save_favorability()
            logger.info("插件卸载，好感度数据已保存")

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
