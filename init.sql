-- Portfolio Sentiment Analyzer — E4 schema
-- Run once on first startup (handled by docker-compose healthcheck + depends_on)

CREATE TABLE IF NOT EXISTS analyses (
    id                       SERIAL PRIMARY KEY,
    user_token_hash          VARCHAR(8)   NOT NULL,
    created_at               TIMESTAMP    NOT NULL DEFAULT NOW(),
    tickers                  TEXT[]       NOT NULL,
    results                  JSONB        NOT NULL,
    analysis_duration_seconds INTEGER
);

-- GIN index — fast lookups like: WHERE results @> '[{"ticker":"AAPL"}]'
CREATE INDEX IF NOT EXISTS idx_analyses_results ON analyses USING GIN (results);

-- Fast per-user history queries
CREATE INDEX IF NOT EXISTS idx_analyses_user ON analyses (user_token_hash, created_at DESC);
