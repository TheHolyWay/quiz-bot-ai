import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio

import quiz  # The main bot logic

class TestQuizBot(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Patch OpenAI client, Telegram bot, and environment variables
        self.patcher_openai = patch('quiz.OpenAI', autospec=True)
        self.mock_openai = self.patcher_openai.start()
        self.addCleanup(self.patcher_openai.stop)
        quiz.OPENAI_TOKEN = "dummy"
        quiz.TELEGRAM_BOT_TOKEN = "dummy"

        # Patch os.getenv
        self.patcher_getenv = patch('quiz.os.getenv', side_effect=lambda k, d=None: d or "dummy")
        self.patcher_getenv.start()
        self.addCleanup(self.patcher_getenv.stop)

        # Patch stats file IO
        self.patcher_exists = patch('quiz.os.path.exists', return_value=True)
        self.patcher_exists.start()
        self.addCleanup(self.patcher_exists.stop)

        self.patcher_open = patch('quiz.open', unittest.mock.mock_open(read_data='{"123": {"name": "Test", "prizes_won": 1, "quizzes_played": 2, "correct_answers_total": 5}}'), create=True)
        self.patcher_open.start()
        self.addCleanup(self.patcher_open.stop)

        self.patcher_json_load = patch('quiz.json.load', return_value={"123": {"name": "Test", "prizes_won": 1, "quizzes_played": 2, "correct_answers_total": 5}})
        self.patcher_json_load.start()
        self.addCleanup(self.patcher_json_load.stop)

        self.patcher_json_dump = patch('quiz.json.dump')
        self.patcher_json_dump.start()
        self.addCleanup(self.patcher_json_dump.stop)

        self.patcher_replace = patch('quiz.os.replace')
        self.patcher_replace.start()
        self.addCleanup(self.patcher_replace.stop)

    def test_load_stats(self):
        stats = quiz.load_stats()
        self.assertEqual(stats["123"]["name"], "Test")

    async def test_save_stats(self):
        stats = {"456": {"name": "Player"}}
        await quiz.save_stats(stats)
        quiz.json.dump.assert_called()

    def test__to_int(self):
        self.assertEqual(quiz._to_int("12"), 12)
        self.assertEqual(quiz._to_int("bad", 99), 99)

    async def test_bump_player_stats(self):
        await quiz.bump_player_stats(user_id=999, name="Tester", inc_prizes=1, inc_correct=2, inc_quizzes=1)
        quiz.json.dump.assert_called()

    def test_difficulty_for(self):
        self.assertEqual(quiz.difficulty_for(0), "easy")
        self.assertEqual(quiz.difficulty_for(5), "medium")
        self.assertEqual(quiz.difficulty_for(8), "hard")

    def test_safe_trim(self):
        # The current implementation trims to limit-1 and then adds "…"
        self.assertEqual(quiz._safe_trim("test", 2), "t…")
        self.assertEqual(quiz._safe_trim("short", 10), "short")

    def test_sanitize_options(self):
        opt = ["A", "B", "B", "", None]
        cleaned = quiz._sanitize_options(opt)
        self.assertEqual(len(cleaned), 4)
        self.assertEqual(cleaned[0], "A")
        self.assertEqual(cleaned[1], "B")

    def test_shuffle_options_keep_answer(self):
        opts = ["A", "B", "C", "D"]
        shuffled, idx = quiz._shuffle_options_keep_answer(opts, 2)
        self.assertEqual(set(shuffled), set(opts))
        self.assertTrue(0 <= idx < 4)

    async def test_decide_theme_from_args(self):
        theme, mixed = await quiz.decide_theme_from_args(["История"])
        self.assertEqual(theme, "История")
        self.assertFalse(mixed)
        with patch('quiz._chat_text', return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="Микс"))])):
            theme, mixed = await quiz.decide_theme_from_args([])
            self.assertTrue(theme)

    async def test_generate_prize_for_theme(self):
        with patch('quiz._chat_text', return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="Приз"))])):
            prize = await quiz.generate_prize_for_theme("Тест")
            self.assertTrue("Приз" in prize or "Тест" in prize)

    async def test_generate_topic_plan_via_llm(self):
        with patch('quiz._chat_json', return_value=MagicMock(choices=[MagicMock(message=MagicMock(content='{"topics":["A","B","C","D","E","F","G","H","I","J"]}'))])):
            topics = await quiz.generate_topic_plan_via_llm("Микс", mixed=True)
            self.assertEqual(len(topics), quiz.QUESTION_COUNT)
            self.assertIn("A", topics)

    async def test_generate_question(self):
        with patch('quiz._chat_json', side_effect=[
            MagicMock(choices=[MagicMock(message=MagicMock(content='{"question":"Q?","options":["A","B","C","D"],"correct_index":1,"explanation":"Пояснение"}'))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content='{"correct_index":1,"explanation":"OK"}'))]),
        ]):
            q = await quiz.generate_question("Тема", 0, "Подтема", [])
            self.assertEqual(q.correct_index, 1)
            self.assertTrue(q.text)

    async def test_cmd_start_and_statistics(self):
        context = MagicMock()
        context.args = []
        context.bot.send_message = AsyncMock()
        context.bot.send_poll = AsyncMock()  # <-- FIXED!
        context.application.bot_data = {}
        update = MagicMock()
        update.effective_chat.id = 555
        await quiz.cmd_start(update, context)
        await quiz.cmd_statistics(update, context)
        context.bot.send_message.assert_called()

    async def test_ask_next_question(self):
        context = MagicMock()
        context.application.bot_data = {
            "quizzes": {
                123: quiz.QuizState(theme="M", prize="P", topic_plan=["T"] * quiz.QUESTION_COUNT)
            }
        }
        context.bot.send_poll = AsyncMock(return_value=MagicMock(poll=MagicMock(id="pollid")))
        await quiz.ask_next_question(context, 123)
        context.bot.send_poll.assert_called()

    async def test_on_poll_answer(self):
        context = MagicMock()
        context.application.bot_data = {
            "poll_to_chat": {"pollid": 123},
            "quizzes": {
                123: quiz.QuizState(
                    theme="T", prize="P", topic_plan=["T"] * quiz.QUESTION_COUNT, current_q=quiz.CurrentQuestion(
                        text="Q", options=["A", "B", "C", "D"], correct_index=0, explanation="E"
                    ), poll_id="pollid"
                )
            }
        }
        pa = MagicMock()
        pa.user = MagicMock(id=1, full_name="F", username="U")
        pa.option_ids = [0]
        pa.poll_id = "pollid"
        update = MagicMock(poll_answer=pa)
        await quiz.on_poll_answer(update, context)
        # Should update scores
        quiz_state = context.application.bot_data["quizzes"][123]
        self.assertEqual(quiz_state.scores[1], 1)

    async def test_finalize_quiz(self):
        context = MagicMock()
        context.bot.send_message = AsyncMock()
        context.application.bot_data = {
            f"name_{1}": "Player1",
            "quizzes": {
                321: quiz.QuizState(
                    theme="T", prize="P", topic_plan=["T"] * quiz.QUESTION_COUNT,
                    scores={1: quiz.PASS_THRESHOLD}, participants={1}
                )
            }
        }
        await quiz.finalize_quiz(context, 321)
        context.bot.send_message.assert_called()

    def test_parse_theme_from_args(self):
        theme = quiz.parse_theme_from_args(["Тест"])
        self.assertEqual(theme, "Тест")
        theme = quiz.parse_theme_from_args([])
        self.assertIsNone(theme)

    def test_ensure_tokens(self):
        quiz.TELEGRAM_BOT_TOKEN = "token"
        quiz.OPENAI_TOKEN = "token"
        quiz.ensure_tokens()  # Should not raise

if __name__ == "__main__":
    unittest.main()