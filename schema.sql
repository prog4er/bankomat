PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
  version INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
  user_id INTEGER PRIMARY KEY AUTOINCREMENT,
  full_name TEXT NOT NULL,
  pin_salt BLOB NOT NULL,
  pin_hash BLOB NOT NULL,
  pin_failed_attempts INTEGER NOT NULL DEFAULT 0,
  is_blocked INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cards (
  card_id INTEGER PRIMARY KEY AUTOINCREMENT,
  card_token TEXT NOT NULL UNIQUE,
  user_id INTEGER NOT NULL,
  is_active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS accounts (
  account_id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  currency TEXT NOT NULL DEFAULT 'RUB',
  balance_cents INTEGER NOT NULL DEFAULT 0,
  daily_withdraw_limit_cents INTEGER NOT NULL DEFAULT 500000,
  daily_withdrawn_cents INTEGER NOT NULL DEFAULT 0,
  daily_withdrawn_date TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  card_id INTEGER NOT NULL,
  authenticated_user_id INTEGER,
  state TEXT NOT NULL,
  created_at TEXT NOT NULL,
  last_activity_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  closed_at TEXT,
  FOREIGN KEY (card_id) REFERENCES cards(card_id),
  FOREIGN KEY (authenticated_user_id) REFERENCES users(user_id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(authenticated_user_id);

CREATE TABLE IF NOT EXISTS events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_events_session_time ON events(session_id, created_at);

CREATE TABLE IF NOT EXISTS transactions (
  txn_id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  user_id INTEGER NOT NULL,
  account_id INTEGER NOT NULL,
  txn_type TEXT NOT NULL,              -- BALANCE|WITHDRAW|DEPOSIT|TRANSFER_OUT|TRANSFER_IN
  amount_cents INTEGER NOT NULL,
  status TEXT NOT NULL,                -- APPROVED|DECLINED
  balance_before_cents INTEGER NOT NULL,
  balance_after_cents INTEGER NOT NULL,
  counterparty_account_id INTEGER,
  created_at TEXT NOT NULL,
  ml_model TEXT,
  ml_risk REAL,
  ml_flagged INTEGER NOT NULL DEFAULT 0,
  features_json TEXT NOT NULL DEFAULT '{}',
  label_is_suspicious INTEGER,          -- только для синтетики/обучения
  FOREIGN KEY (session_id) REFERENCES sessions(session_id),
  FOREIGN KEY (user_id) REFERENCES users(user_id),
  FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

CREATE INDEX IF NOT EXISTS idx_txn_user_time ON transactions(user_id, created_at);
