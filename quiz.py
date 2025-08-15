# quiz_poll_bot_mixed_llm.py
# -*- coding: utf-8 -*-
"""
Telegram quiz-bot using real Polls of type 'quiz' — LLM-driven Mixed Trivia.

Fixes:
- Shuffle options with correct_index remap (правильный не всегда A).
- Авто-итог по таймауту и автопереход через 5 секунд (без сообщения "Время истекло" и без кнопки).
- finalize_quiz вызывается корректно после 10-го вопроса, выводит табло.
- Статистика устойчива к старому формату json (добавляет недостающие ключи).
- Призы считаются один раз за квиз. Идемпотентность финала через флаг finished.
- Трекинг имени пользователя — по вашему лямбда-сниппету.

Env:
- TELEGRAM_BOT_TOKEN, OPENAI_TOKEN
- QUIZ_OPENAI_MODEL     (default: gpt-4o)
- QUIZ_OPENAI_VERIFIER  (default: gpt-4o-mini)
"""

import asyncio
import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from telegram import Update
from telegram.ext import (
  ApplicationBuilder,
  CommandHandler,
  ContextTypes,
  PollAnswerHandler,
  MessageHandler,
  filters,
)

# ==========================
# Configuration and Globals
# ==========================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_TOKEN = os.getenv("OPENAI_TOKEN")

STATS_FILE = "quiz_stats.json"
QUESTION_COUNT = 10
PASS_THRESHOLD = 7

# Time controls
OPEN_PERIOD_SECONDS = 30        # время на ответ
FIRST_COUNTDOWN_SECONDS = 5     # обратный отсчёт перед первым вопросом

# Models
GEN_MODEL = os.getenv("QUIZ_OPENAI_MODEL", "gpt-4o")
VER_MODEL = os.getenv("QUIZ_OPENAI_VERIFIER", "gpt-4o-mini")  # gpt-4.1 если доступен

# OpenAI client
if not OPENAI_TOKEN:
  raise RuntimeError("OPENAI_TOKEN is not set in environment.")
client = OpenAI(api_key=OPENAI_TOKEN)

# Single global lock for stats file IO
stats_lock = asyncio.Lock()

# ==========================
# Persistence (stats)
# ==========================

def load_stats() -> Dict[str, dict]:
  if not os.path.exists(STATS_FILE):
    return {}
  try:
    with open(STATS_FILE, "r", encoding="utf-8") as f:
      return json.load(f)
  except Exception:
    return {}

async def save_stats(stats: Dict[str, dict]) -> None:
  async with stats_lock:
    tmp = STATS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
      json.dump(stats, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATS_FILE)

def _to_int(x, default=0) -> int:
  try:
    return int(x)
  except Exception:
    return default

async def bump_player_stats(
    user_id: int,
    name: str,
    inc_prizes: int = 0,
    inc_correct: int = 0,
    inc_quizzes: int = 0,
) -> None:
  """Robust updates even if old stats file lacks some keys."""
  stats = load_stats()
  key = str(user_id)
  row = stats.get(key) or {}
  # ensure keys exist and are ints
  row["name"] = name
  row["prizes_won"] = _to_int(row.get("prizes_won", 0))
  row["quizzes_played"] = _to_int(row.get("quizzes_played", 0))
  row["correct_answers_total"] = _to_int(row.get("correct_answers_total", 0))
  # apply increments
  row["prizes_won"] += int(inc_prizes)
  row["quizzes_played"] += int(inc_quizzes)
  row["correct_answers_total"] += int(inc_correct)
  stats[key] = row
  await save_stats(stats)

# ==========================
# Quiz state model
# ==========================

@dataclass
class CurrentQuestion:
  text: str
  options: List[str]  # len=4
  correct_index: int  # 0..3
  explanation: str

@dataclass
class QuizState:
  theme: str
  prize: str
  topic_plan: List[str] = field(default_factory=list)    # 10 подтем
  current_index: int = 0  # 0..QUESTION_COUNT-1
  scores: Dict[int, int] = field(default_factory=dict)  # user_id -> correct count
  participants: set = field(default_factory=set)        # все, кто отвечал хоть раз
  current_q: Optional[CurrentQuestion] = None
  poll_id: Optional[str] = None
  next_enabled: bool = False                            # флаг "итог выведен"
  answers: Dict[int, int] = field(default_factory=dict) # user_id -> chosen index
  answered_users: set = field(default_factory=set)      # чтобы не учитывать повторно
  asked_texts: set = field(default_factory=set)         # контроль повторов
  finished: bool = False                                # идемпотентность финала

# ==========================
# Difficulty helpers
# ==========================

def difficulty_for(question_idx: int) -> str:
  if question_idx < 3:
    return "easy"
  if question_idx < 7:
    return "medium"
  return "hard"

# ==========================
# OpenAI helpers
# ==========================

def _chat_json(model: str, system: str, user: str, temperature: float = 0.7, max_tokens: int = 500):
  return client.chat.completions.create(
      model=model,
      messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user},
      ],
      response_format={"type": "json_object"},
      temperature=temperature,
      max_tokens=max_tokens,
  )

def _chat_text(model: str, system: str, user: str, temperature: float = 0.7, max_tokens: int = 200):
  return client.chat.completions.create(
      model=model,
      messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user},
      ],
      temperature=temperature,
      max_tokens=max_tokens,
  )

def _safe_trim(s: str, limit: int) -> str:
  s = (s or "").strip()
  return (s[: limit - 1] + "…") if len(s) > limit else s

def _sanitize_options(options: List[str]) -> List[str]:
  seen = set()
  cleaned = []
  for o in options:
    t = _safe_trim(str(o), 100)
    if not t:
      continue
    key = t.lower()
    if key in seen:
      continue
    seen.add(key)
    cleaned.append(t)
  while len(cleaned) < 4:
    cleaned.append(f"Вариант {len(cleaned)+1}")
  return cleaned[:4]

def _shuffle_options_keep_answer(options: List[str], correct_index: int) -> Tuple[List[str], int]:
  order = list(range(len(options)))
  random.shuffle(order)
  shuffled = [options[i] for i in order]
  new_correct = order.index(correct_index)
  return shuffled, new_correct

# ==========================
# LLM-driven Theme & Plan
# ==========================

async def decide_theme_from_args(args: List[str]) -> Tuple[str, bool]:
  if args:
    theme = " ".join(a for a in args if a).strip()
    return theme or "Смешанная викторина", False

  sys = "Ты придумываешь короткие названия для квизов на русском."
  usr = (
    "Сгенерируй краткое название (до 40 символов) для СМЕШАННОЙ викторины, "
    "где вопросы идут из разных областей. Без эмодзи, одна строка."
  )
  try:
    r = await asyncio.to_thread(_chat_text, GEN_MODEL, sys, usr, 0.8, 60)
    theme = (r.choices[0].message.content or "").strip().replace("\n", " ")
    theme = _safe_trim(theme, 40) or "Смешанная викторина"
  except Exception:
    theme = "Смешанная викторина"
  return theme, True

async def generate_prize_for_theme(theme: str) -> str:
  sys = "Ты придумываешь короткие и уместные призы/бейджи для квизов на русском."
  usr = f"""
Тема квиза: {theme}
Сформулируй очень короткий приз (до 40 символов), без эмодзи, в одной строке.
Стиль: вроде "Значок знатока", "Орден викторины", "Медаль квизёра".
Верни только текст.
"""
  try:
    r = await asyncio.to_thread(_chat_text, GEN_MODEL, sys, usr, 0.8, 60)
    prize = (r.choices[0].message.content or "").strip().replace("\n", " ")
    prize = _safe_trim(prize, 40)
    return prize or f"Знак отличия: {theme}"
  except Exception:
    return f"Знак отличия: {theme}"

async def generate_topic_plan_via_llm(theme: str, mixed: bool) -> List[str]:
  sys = ("Ты составляешь план квиза (10 подтем) на русском. Верни строго JSON с ключом 'topics' — массив из 10 строк.")
  scope = (
    "Смешанная викторина: включи разнообразные области — общество, история, география, наука, техника, культура, литература, фильмы/сериалы, музыка, спорт."
    if mixed else
    "Однотематическая викторина: выбери 10 разных подтем, реально относящихся к теме."
  )
  usr = f"""
Тема: {theme}
Задача: {scope}
Требования:
- Подтемы самодостаточные, без нумерации.
- Исключи повторы.
- Длина каждой до 60 символов.
Формат строго:
{{"topics": ["подтема1","подтема2","...","подтема10"]}}
"""
  try:
    r = await asyncio.to_thread(_chat_json, GEN_MODEL, sys, usr, 0.7, 300)
    content = (r.choices[0].message.content or "").strip()
    data = json.loads(content)
    topics = [_safe_trim(str(t), 60) for t in list(data.get("topics", [])) if str(t).strip()]
    seen, unique = set(), []
    for t in topics:
      k = t.lower()
      if k not in seen:
        seen.add(k); unique.append(t)
    while len(unique) < QUESTION_COUNT:
      unique.append(f"Случайная тема {len(unique)+1}")
    return unique[:QUESTION_COUNT]
  except Exception:
    return [f"Случайная тема {i+1}" for i in range(QUESTION_COUNT)]

# ==========================
# Generation & verification
# ==========================

async def generate_question(theme: str, question_idx: int, topic: str, avoid_phrases: List[str]) -> CurrentQuestion:
  diff = difficulty_for(question_idx)
  nonce = os.urandom(4).hex()

  sys_gen = (
    "Ты генератор вопросов для викторины на русском. "
    "Пиши чётко и нейтрально; допускается лёгкий юмор без оскорблений. "
    "Варианты правдоподобные; не более одного шуточного."
  )
  avoid_block = ""
  if avoid_phrases:
    clipped = [("- " + _safe_trim(p, 120)) for p in avoid_phrases[-5:]]
    avoid_block = "Не повторяй и не перефразируй эти формулировки:\n" + "\n".join(clipped) + "\n"

  user_gen = f"""
Тема квиза: {theme}
Подтема/категория: {topic}
Сложность: {diff}
Стиль: обычная викторина (вопрос одной строкой), без спойлеров и двусмысленностей.

{avoid_block}
Сгенерируй РОВНО ОДИН вопрос. Верни СТРОГО JSON:
{{
  "question": "краткий ясный вопрос (уникальная формулировка)",
  "options": ["вариант A","вариант B","вариант C","вариант D"],
  "correct_index": 0,
  "explanation": "почему правильно (до 180 символов)"
}}
Ограничения:
- Ровно 4 варианта. Один — правильный. Остальные — правдоподобные, но неверные.
- Градиент сложности: easy -> medium -> hard.
- Никаких выдумок.
[nonce:{nonce}]
"""

  try:
    resp_gen = await asyncio.to_thread(_chat_json, GEN_MODEL, sys_gen, user_gen, 0.85, 450)
    gen_text = (resp_gen.choices[0].message.content or "").strip()
    data = json.loads(gen_text)
    q_text = _safe_trim(str(data.get("question", "")), 240)
    options = _sanitize_options(list(data.get("options", [])))
    correct_idx = int(data.get("correct_index", 0))
    explanation = _safe_trim(str(data.get("explanation", "")), 190)
    if not (0 <= correct_idx < 4) or not q_text or len(options) != 4:
      raise ValueError("Bad generation schema")
  except Exception:
    q_text = f"{theme}: выберите верный вариант."
    options = ["Самый очевидный ответ", "Правильный ответ", "Очень заманчивый вариант", "Случайный выбор"]
    correct_idx = 1
    explanation = "Правильный вариант соответствует общепризнанным фактам."

  sys_ver = (
    "Ты строгий верификатор квиз-вопросов. Определи ОДИН корректный индекс (0..3) и кратко объясни."
  )
  user_ver = f"""
Вопрос: {q_text}
Варианты (индексированы 0..3):
0) {options[0]}
1) {options[1]}
2) {options[2]}
3) {options[3]}

Контекст:
- Тема: {theme}
- Подтема: {topic}
- Сложность: {diff}

Верни строго JSON:
{{ "correct_index": 0, "explanation": "краткая причина (<=160 символов)" }}
[nonce:{nonce}]
"""
  try:
    resp_ver = await asyncio.to_thread(_chat_json, VER_MODEL, sys_ver, user_ver, 0.2, 220)
    ver_text = (resp_ver.choices[0].message.content or "").strip()
    ver = json.loads(ver_text)
    v_idx = int(ver.get("correct_index", correct_idx))
    v_exp = _safe_trim(str(ver.get("explanation", explanation)), 190)
    if 0 <= v_idx < 4:
      correct_idx = v_idx
      explanation = v_exp
  except Exception:
    pass

  return CurrentQuestion(text=q_text, options=options, correct_index=correct_idx, explanation=explanation)

# ==========================
# Async utilities (no JobQueue)
# ==========================

async def countdown_then(func, *args, seconds: int = FIRST_COUNTDOWN_SECONDS, **kwargs):
  context: ContextTypes.DEFAULT_TYPE = kwargs.get("context") or args[0]
  chat_id: int = kwargs.get("chat_id") or args[1]

  if seconds <= 0:
    return await func(*args, **kwargs)

  try:
    msg = await context.bot.send_message(chat_id=chat_id, text=f"Старт через {seconds}…")
    for t in range(seconds - 1, 0, -1):
      await asyncio.sleep(1)
      try:
        await context.bot.edit_message_text(chat_id=chat_id, message_id=msg.message_id, text=f"Старт через {t}…")
      except Exception:
        await context.bot.send_message(chat_id=chat_id, text=f"Старт через {t}…")
    await asyncio.sleep(1)
    try:
      await context.bot.edit_message_text(chat_id=chat_id, message_id=msg.message_id, text="Погнали!")
    except Exception:
      await context.bot.send_message(chat_id=chat_id, text="Погнали!")
  except Exception:
    pass

  await func(*args, **kwargs)

async def schedule_auto_advance(context: ContextTypes.DEFAULT_TYPE, chat_id: int, q_index: int, delay: int) -> None:
  """По таймауту показать итог вопроса и перейти дальше через 5 секунд."""
  await asyncio.sleep(delay)

  quizzes: Dict[int, QuizState] = context.application.bot_data.get("quizzes", {})
  quiz: Optional[QuizState] = quizzes.get(chat_id)
  if not quiz or quiz.finished or quiz.current_index != q_index or not quiz.current_q:
    return

  # Пер-вопросный итог
  letters = ["A", "B", "C", "D"]
  correct_letter = letters[quiz.current_q.correct_index]
  winners = []
  for uid, chosen in quiz.answers.items():
    if chosen == quiz.current_q.correct_index:
      name = context.application.bot_data.get(f"name_{uid}", f"Игрок {uid}")
      winners.append(name)
  winners_text = ", ".join(winners) if winners else "Никто не попал."

  try:
    await context.bot.send_message(
        chat_id=chat_id,
        text=(
          f"Правильный ответ: {correct_letter}\n"
          f"Пояснение: {quiz.current_q.explanation}\n"
          f"Кто ответил верно: {winners_text}"
        )
    )
  except Exception:
    pass

  quiz.next_enabled = True
  quiz.current_index += 1

  # Если вопросы закончились — финализация; иначе следующий вопрос
  if quiz.current_index >= QUESTION_COUNT:
    await finalize_quiz(context, chat_id)
    return

  await asyncio.sleep(5)
  await ask_next_question(context, chat_id)

# ==========================
# Bot Handlers
# ==========================

def parse_theme_from_args(args: List[str]) -> Optional[str]:
  if not args:
    return None
  theme = " ".join(a for a in args if a).strip()
  return theme or None

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  chat_id = update.effective_chat.id

  user_theme = parse_theme_from_args(context.args)
  if user_theme:
    theme, is_mixed = (user_theme, False)
  else:
    theme, is_mixed = await decide_theme_from_args([])

  prize = await generate_prize_for_theme(theme)
  plan = await generate_topic_plan_via_llm(theme, mixed=is_mixed)

  quizzes: Dict[int, QuizState] = context.application.bot_data.setdefault("quizzes", {})
  quizzes[chat_id] = QuizState(theme=theme, prize=prize, topic_plan=plan)

  await update.message.reply_text(
      f"Тема квиза: {theme}\n"
      f"Приз: {prize}\n\n"
      f"Всего {QUESTION_COUNT} вопросов. Нужно {PASS_THRESHOLD}+ правильных, чтобы забрать приз.\n"
      f"На каждый вопрос даётся минимум {OPEN_PERIOD_SECONDS} сек. Ответы не анонимные."
  )

  await countdown_then(ask_next_question, context, chat_id, seconds=FIRST_COUNTDOWN_SECONDS)

async def cmd_statistics(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  stats = load_stats()
  if not stats:
    await update.message.reply_text("Статистика пуста.")
    return
  rows = sorted(
      stats.values(),
      key=lambda x: (x.get("prizes_won", 0), x.get("correct_answers_total", 0)),
      reverse=True,
  )[:15]
  lines = ["Топ игроков:"]
  for i, r in enumerate(rows):
    lines.append(
        f"{i+1}. {r.get('name','Игрок')} — призов: {r.get('prizes_won',0)}, "
        f"правильных ответов: {r.get('correct_answers_total',0)}, "
        f"квизов сыграно: {r.get('quizzes_played',0)}"
    )
  await update.message.reply_text("\n".join(lines))

async def ask_next_question(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
  quizzes: Dict[int, QuizState] = context.application.bot_data.get("quizzes", {})
  quiz: Optional[QuizState] = quizzes.get(chat_id)
  if not quiz or quiz.finished:
    return

  if quiz.current_index >= QUESTION_COUNT:
    await finalize_quiz(context, chat_id)
    return

  topic = quiz.topic_plan[quiz.current_index] if quiz.topic_plan else "Случайная категория"

  # Пытаемся избежать повторов формулировок
  attempts = 0
  q: Optional[CurrentQuestion] = None
  avoid = list(quiz.asked_texts)[-5:]
  candidate: Optional[CurrentQuestion] = None
  while attempts < 3:
    candidate = await generate_question(theme=quiz.theme, question_idx=quiz.current_index, topic=topic, avoid_phrases=avoid)
    norm = candidate.text.lower().strip()
    if norm not in quiz.asked_texts:
      q = candidate
      quiz.asked_texts.add(norm)
      break
    attempts += 1
    avoid.append(candidate.text)
  if q is None:
    q = candidate

  # Перемешиваем варианты, чтобы правильный не был всегда A
  q.options, q.correct_index = _shuffle_options_keep_answer(q.options, q.correct_index)

  quiz.current_q = q
  quiz.answers.clear()
  quiz.answered_users.clear()
  quiz.next_enabled = False

  sent_poll = await context.bot.send_poll(
      chat_id=chat_id,
      question=f"Вопрос {quiz.current_index + 1}/{QUESTION_COUNT} — {q.text}",
      options=q.options,
      is_anonymous=False,
      type="quiz",
      correct_option_id=q.correct_index,
      explanation=q.explanation,
      open_period=OPEN_PERIOD_SECONDS,
      allows_multiple_answers=False,
  )
  quiz.poll_id = sent_poll.poll.id

  # Привязываем poll_id -> chat_id
  poll_map: Dict[str, int] = context.application.bot_data.setdefault("poll_to_chat", {})
  poll_map[sent_poll.poll.id] = chat_id

  # Авто-переход без кнопки и без сообщения "Время истекло"
  asyncio.create_task(
      schedule_auto_advance(
          context=context,
          chat_id=chat_id,
          q_index=quiz.current_index,
          delay=OPEN_PERIOD_SECONDS,
      )
  )

async def on_poll_answer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  pa = update.poll_answer
  if not pa:
    return
  user = pa.user
  option_ids = pa.option_ids or []
  chosen = option_ids[0] if option_ids else None
  poll_id = pa.poll_id

  # chat по poll_id
  poll_map: Dict[str, int] = context.application.bot_data.get("poll_to_chat", {})
  chat_id = poll_map.get(poll_id)
  if chat_id is None:
    return

  quizzes: Dict[int, QuizState] = context.application.bot_data.get("quizzes", {})
  quiz: Optional[QuizState] = quizzes.get(chat_id)
  if not quiz or not quiz.current_q or quiz.poll_id != poll_id or quiz.finished:
    return

  # Запоминаем display name для статистики (ваш сниппет добавлен в bootstrap, но не мешает продублировать)
  if user:
    display = (user.full_name or user.username or str(user.id)).strip()
    context.application.bot_data[f"name_{user.id}"] = display

  if chosen is None or user is None:
    return
  if user.id in quiz.answered_users:
    return

  quiz.answers[user.id] = chosen
  quiz.answered_users.add(user.id)
  quiz.participants.add(user.id)  # фикс: считаем всех участников квиза

  if chosen == quiz.current_q.correct_index:
    quiz.scores[user.id] = quiz.scores.get(user.id, 0) + 1
    await bump_player_stats(
        user_id=user.id,
        name=context.application.bot_data.get(f"name_{user.id}", f"Игрок {user.id}"),
        inc_correct=1,
    )

async def finalize_quiz(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
  quizzes: Dict[int, QuizState] = context.application.bot_data.get("quizzes", {})
  quiz: Optional[QuizState] = quizzes.get(chat_id)
  if not quiz or quiz.finished:
    return
  quiz.finished = True  # идемпотентность

  # Победители (>= PASS_THRESHOLD)
  winners = sorted(uid for uid, score in quiz.scores.items() if score >= PASS_THRESHOLD)

  # Участники — все, кто хоть раз отвечал
  participants = set(quiz.participants)

  # Сохраняем сыгранные квизы всем участникам
  for uid in participants:
    await bump_player_stats(
        user_id=uid,
        name=context.application.bot_data.get(f"name_{uid}", f"Игрок {uid}"),
        inc_quizzes=1,
    )

  # Сохраняем призы победителям (один раз за квиз)
  for uid in winners:
    await bump_player_stats(
        user_id=uid,
        name=context.application.bot_data.get(f"name_{uid}", f"Игрок {uid}"),
        inc_prizes=1,
    )

  # Рендер табло
  if quiz.scores:
    lines = [f"Итоги квиза: {quiz.theme} — приз: {quiz.prize}"]
    top = sorted(quiz.scores.items(), key=lambda kv: kv[1], reverse=True)
    for uid, sc in top:
      nm = context.application.bot_data.get(f"name_{uid}", f"Игрок {uid}")
      lines.append(f"{nm}: {sc}/{QUESTION_COUNT}")
    if winners:
      got = ", ".join(context.application.bot_data.get(f"name_{uid}", f'Игрок {uid}') for uid in winners)
      lines.append("")
      lines.append(f"Приз получают: {got}")
    else:
      lines.append("")
      lines.append("Сегодня без призов. Попробуем ещё?")
  else:
    lines = ["Никто не поучаствовал. Запускаем ещё раз?"]

  lines.append("\nЧтобы начать заново, отправьте /start или /start <тема>.")

  try:
    await context.bot.send_message(chat_id=chat_id, text="\n".join(lines))
  finally:
    # Cleanup
    quizzes.pop(chat_id, None)

# ==========================
# Error handler
# ==========================

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
  import traceback
  err = "".join(traceback.format_exception(None, context.error, context.error.__traceback__))
  print("ERROR:", err)
  try:
    chat_id = getattr(getattr(update, "effective_chat", None), "id", None)
    if chat_id:
      await context.bot.send_message(
          chat_id,
          "Упс, что-то сломалось. Переходим дальше. Используйте /start для нового квиза."
      )
  except Exception:
    pass

# ==========================
# App bootstrap
# ==========================

def ensure_tokens():
  if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in environment.")
  if not OPENAI_TOKEN:
    raise RuntimeError("OPENAI_TOKEN is not set in environment.")

def main():
  ensure_tokens()
  app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

  # ваш сниппет трекинга имени:
  app.add_handler(
      MessageHandler(filters.ALL, lambda u, c: (
        c.application.bot_data.__setitem__(
            f"name_{getattr(getattr(u, 'effective_user', None), 'id', 'x')}",
            (getattr(getattr(u, 'effective_user', None), 'full_name', None)
             or getattr(getattr(u, 'effective_user', None), 'username', None)
             or str(getattr(getattr(u, 'effective_user', None), 'id', 'x')))
        )
      )),
      group=-1
  )

  # Команды
  app.add_handler(CommandHandler("start", cmd_start))
  app.add_handler(CommandHandler("statistics", cmd_statistics))

  # Ответы на опросы
  app.add_handler(PollAnswerHandler(on_poll_answer))

  # Ошибки
  app.add_error_handler(error_handler)

  app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
  main()