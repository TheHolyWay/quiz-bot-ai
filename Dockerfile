FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on

RUN apt-get update && apt-get install -y --no-install-recommends tzdata && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy main source and test files
COPY quiz.py /app/quiz.py
COPY test_quiz.py /app/test_quiz.py

RUN chmod 777 -R /app

# Set dummy envs for test run (can be overridden later)
ENV TELEGRAM_BOT_TOKEN=dummy
ENV OPENAI_TOKEN=dummy

# --- Run tests during build; fail build if tests fail ---
RUN python -m pip install pytest pytest-asyncio
RUN pytest /app/test_quiz.py
RUN rm /app/test_quiz.py

# Default command: run bot (after build completes)
CMD ["python", "-u", "/app/quiz.py"]