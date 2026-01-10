-- ==========================================================
-- LLM Observatory Database Schema
-- ==========================================================
-- Run this in your Supabase SQL Editor to set up the database
-- This creates all tables, indexes, and helper functions.

-- =====================================================
-- CORE TABLES
-- =====================================================

-- Metrics table for storing LLM API call data
CREATE TABLE IF NOT EXISTS metrics (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    model TEXT NOT NULL,
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    latency_ms FLOAT NOT NULL DEFAULT 0,
    cost FLOAT NOT NULL DEFAULT 0,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    error TEXT,
    endpoint TEXT,
    project_id UUID,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Projects table for organizing metrics
CREATE TABLE IF NOT EXISTS projects (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add foreign key if not exists
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'metrics_project_id_fkey'
    ) THEN
        ALTER TABLE metrics 
        ADD CONSTRAINT metrics_project_id_fkey 
        FOREIGN KEY (project_id) REFERENCES projects(id);
    END IF;
END $$;

-- =====================================================
-- INDEXES
-- =====================================================

CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_model ON metrics(model);
CREATE INDEX IF NOT EXISTS idx_metrics_endpoint ON metrics(endpoint);
CREATE INDEX IF NOT EXISTS idx_metrics_project ON metrics(project_id);

-- =====================================================
-- ROW LEVEL SECURITY
-- =====================================================

ALTER TABLE metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;

-- Create policies (service role has full access)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies 
        WHERE schemaname = 'public' 
          AND tablename = 'metrics' 
          AND policyname = 'Service role has full access'
    ) THEN
        CREATE POLICY "Service role has full access" ON metrics
            FOR ALL USING (true) WITH CHECK (true);
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies 
        WHERE schemaname = 'public' 
          AND tablename = 'projects' 
          AND policyname = 'Service role has full access to projects'
    ) THEN
        CREATE POLICY "Service role has full access to projects" ON projects
            FOR ALL USING (true) WITH CHECK (true);
    END IF;
END $$;

-- =====================================================
-- HELPER FUNCTIONS
-- =====================================================

-- Get aggregated stats with optional filters
CREATE OR REPLACE FUNCTION get_metrics_stats(
    start_time TIMESTAMPTZ DEFAULT NULL,
    end_time TIMESTAMPTZ DEFAULT NULL,
    filter_project_id UUID DEFAULT NULL
)
RETURNS JSON AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'total_cost', COALESCE(SUM(cost), 0),
        'total_tokens_in', COALESCE(SUM(tokens_in), 0),
        'total_tokens_out', COALESCE(SUM(tokens_out), 0),
        'total_requests', COUNT(*),
        'avg_latency_ms', COALESCE(AVG(latency_ms), 0),
        'p95_latency_ms', COALESCE(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms), 0),
        'error_count', COUNT(*) FILTER (WHERE error IS NOT NULL),
        'cost_by_model', (
            SELECT COALESCE(json_object_agg(model, total_cost), '{}'::json)
            FROM (SELECT model, SUM(cost) as total_cost FROM metrics 
                  WHERE (start_time IS NULL OR timestamp >= start_time)
                    AND (end_time IS NULL OR timestamp <= end_time)
                    AND (filter_project_id IS NULL OR project_id = filter_project_id)
                    AND error IS NULL
                  GROUP BY model) t
        ),
        'requests_by_model', (
            SELECT COALESCE(json_object_agg(model, cnt), '{}'::json)
            FROM (SELECT model, COUNT(*) as cnt FROM metrics 
                  WHERE (start_time IS NULL OR timestamp >= start_time)
                    AND (end_time IS NULL OR timestamp <= end_time)
                    AND (filter_project_id IS NULL OR project_id = filter_project_id)
                    AND error IS NULL
                  GROUP BY model) t
        )
    ) INTO result
    FROM metrics
    WHERE (start_time IS NULL OR timestamp >= start_time)
      AND (end_time IS NULL OR timestamp <= end_time)
      AND (filter_project_id IS NULL OR project_id = filter_project_id);
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Get latency distribution percentiles per model
CREATE OR REPLACE FUNCTION get_latency_distribution(
    filter_model TEXT DEFAULT NULL,
    filter_project_id UUID DEFAULT NULL,
    start_time TIMESTAMPTZ DEFAULT NULL,
    end_time TIMESTAMPTZ DEFAULT NULL
)
RETURNS JSON AS $$
BEGIN
    RETURN (
        SELECT COALESCE(json_agg(row_to_json(t)), '[]'::json)
        FROM (
            SELECT 
                model,
                COUNT(*) as request_count,
                ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms)::numeric, 2) as p50,
                ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY latency_ms)::numeric, 2) as p75,
                ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY latency_ms)::numeric, 2) as p90,
                ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms)::numeric, 2) as p95,
                ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms)::numeric, 2) as p99,
                ROUND(MIN(latency_ms)::numeric, 2) as min,
                ROUND(MAX(latency_ms)::numeric, 2) as max,
                ROUND(AVG(latency_ms)::numeric, 2) as avg
            FROM metrics
            WHERE (filter_model IS NULL OR model = filter_model)
              AND (filter_project_id IS NULL OR project_id = filter_project_id)
              AND (start_time IS NULL OR timestamp >= start_time)
              AND (end_time IS NULL OR timestamp <= end_time)
              AND error IS NULL
            GROUP BY model
            ORDER BY p95 DESC
        ) t
    );
END;
$$ LANGUAGE plpgsql;

-- Get provider-level aggregated stats
CREATE OR REPLACE FUNCTION get_provider_stats(
    filter_project_id UUID DEFAULT NULL,
    start_time TIMESTAMPTZ DEFAULT NULL,
    end_time TIMESTAMPTZ DEFAULT NULL
)
RETURNS JSON AS $$
BEGIN
    RETURN (
        SELECT COALESCE(json_agg(row_to_json(t)), '[]'::json)
        FROM (
            SELECT 
                CASE 
                    WHEN model LIKE 'gpt%' THEN 'openai'
                    WHEN model LIKE 'claude%' THEN 'anthropic'
                    WHEN model LIKE 'gemini%' THEN 'google'
                    ELSE 'other'
                END as provider,
                COUNT(*) as total_requests,
                ROUND(SUM(cost)::numeric, 4) as total_cost,
                ROUND(AVG(cost)::numeric, 6) as avg_cost_per_call,
                ROUND(AVG(latency_ms)::numeric, 2) as avg_latency,
                ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms)::numeric, 2) as p95_latency,
                SUM(tokens_in) as total_tokens_in,
                SUM(tokens_out) as total_tokens_out,
                COUNT(*) FILTER (WHERE error IS NOT NULL) as error_count
            FROM metrics
            WHERE (filter_project_id IS NULL OR project_id = filter_project_id)
              AND (start_time IS NULL OR timestamp >= start_time)
              AND (end_time IS NULL OR timestamp <= end_time)
            GROUP BY 
                CASE 
                    WHEN model LIKE 'gpt%' THEN 'openai'
                    WHEN model LIKE 'claude%' THEN 'anthropic'
                    WHEN model LIKE 'gemini%' THEN 'google'
                    ELSE 'other'
                END
            ORDER BY total_cost DESC
        ) t
    );
END;
$$ LANGUAGE plpgsql;

-- Get burn rate for real-time cost tracking
CREATE OR REPLACE FUNCTION get_burn_rate(
    filter_project_id UUID DEFAULT NULL,
    window_minutes INTEGER DEFAULT 60
)
RETURNS JSON AS $$
DECLARE
    current_cost FLOAT;
    previous_cost FLOAT;
    burn_rate FLOAT;
BEGIN
    SELECT COALESCE(SUM(cost), 0) INTO current_cost
    FROM metrics
    WHERE timestamp > NOW() - (window_minutes || ' minutes')::INTERVAL
      AND (filter_project_id IS NULL OR project_id = filter_project_id);
    
    SELECT COALESCE(SUM(cost), 0) INTO previous_cost
    FROM metrics
    WHERE timestamp > NOW() - (window_minutes * 2 || ' minutes')::INTERVAL
      AND timestamp <= NOW() - (window_minutes || ' minutes')::INTERVAL
      AND (filter_project_id IS NULL OR project_id = filter_project_id);
    
    burn_rate := current_cost * (60.0 / window_minutes);
    
    RETURN json_build_object(
        'current_window_cost', ROUND(current_cost::numeric, 4),
        'previous_window_cost', ROUND(previous_cost::numeric, 4),
        'hourly_burn_rate', ROUND(burn_rate::numeric, 4),
        'trend', CASE 
            WHEN current_cost > previous_cost * 1.1 THEN 'up'
            WHEN current_cost < previous_cost * 0.9 THEN 'down'
            ELSE 'stable'
        END,
        'trend_percentage', CASE 
            WHEN previous_cost > 0 THEN ROUND(((current_cost - previous_cost) / previous_cost * 100)::numeric, 1)
            ELSE 0
        END
    );
END;
$$ LANGUAGE plpgsql;

