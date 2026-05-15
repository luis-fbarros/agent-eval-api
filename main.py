import os
import uuid
import time
from datetime import datetime
from typing import Optional, List, Dict, Any

import mlflow
import dagshub
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Carrega .env em desenvolvimento local (no container, as vars já vêm pelo ambiente)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from eval_pipeline.worker import evaluate_trace

# ---------------------------------------------------------------------------
# DagsHub / MLflow — autenticação via env var para funcionar em container
# ---------------------------------------------------------------------------
_dagshub_token = os.getenv("DAGSHUB_USER_TOKEN")
if _dagshub_token:
    dagshub.auth.add_app_token(_dagshub_token)

dagshub.init(repo_owner="luis-fbarros", repo_name="MLFlow-agent-eval", mlflow=True)

EXPERIMENT_NAME = "agent-eval-hackathon"
mlflow.set_experiment(EXPERIMENT_NAME)

app = FastAPI(
    title="AI Agent Evaluation Pipeline",
    description="Pipeline for evaluating AI agent traces with DeepEval and MLflow via DagsHub",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TraceAttributes(BaseModel):
    user_prompt: str
    agent_response: str
    agentops_tool_output: Optional[str] = None
    agentops_duration_ms: Optional[float] = None
    gen_ai_usage_input_tokens: Optional[int] = None
    gen_ai_usage_output_tokens: Optional[int] = None

    def to_worker_format(self) -> dict:
        """Converts to the internal expected format by the worker."""
        return {
            "attributes": {
                # Chaves canônicas lidas por syntax_check e static_eval
                "agent_tracer.agent.input": self.user_prompt,
                "agent_tracer.agent.output": self.agent_response,
                "agent_tracer.tool.output": self.agentops_tool_output or "",
                "agent_tracer.duration_ms": self.agentops_duration_ms,
                "gen_ai.usage.input_tokens": self.gen_ai_usage_input_tokens or 0,
                "gen_ai.usage.output_tokens": self.gen_ai_usage_output_tokens or 0,
            }
        }


class IngestPayload(BaseModel):
    trace_id: Optional[str] = None
    run_name: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    attributes: TraceAttributes


class IngestResponse(BaseModel):
    trace_id: str
    status: str
    message: str


class EvalResult(BaseModel):
    trace_id: str
    run_id: str
    run_name: str
    status: str
    start_time: str
    metrics: Dict[str, Any]
    params: Dict[str, Any]
    tags: Dict[str, str]
    mlflow_url: str


class DashboardResponse(BaseModel):
    experiment_name: str
    experiment_id: str
    total_runs: int
    runs: List[EvalResult]


def _run_eval_pipeline(trace_id: str, payload_dict: dict, run_name: str, extra_tags: dict):
    """Executes the complete evaluation and registers in MLflow via DagsHub."""
    json_trace = TraceAttributes(**payload_dict).to_worker_format()
    attrs = json_trace["attributes"]

    with mlflow.start_run(run_name=run_name) as run:
        # Tags
        mlflow.set_tag("trace_id", trace_id)
        mlflow.set_tag("pipeline_version", "1.0.0")
        mlflow.set_tag("evaluated_at", datetime.utcnow().isoformat())
        for k, v in extra_tags.items():
            mlflow.set_tag(k, v)

        mlflow.log_param("user_prompt_length", len(attrs.get("agent_tracer.agent.input") or ""))
        mlflow.log_param("agent_response_length", len(attrs.get("agent_tracer.agent.output") or ""))
        mlflow.log_param("has_tool_output", bool(attrs.get("agent_tracer.tool.output")))

        try:
            result = evaluate_trace(json_trace)
        except Exception as exc:
            mlflow.set_tag("eval_status", "pipeline_error")
            mlflow.log_param("error_message", str(exc)[:500])
            return

        if "syntax_error" in result:
            mlflow.set_tag("eval_status", "syntax_error")
            mlflow.log_param("error_message", result["syntax_error"])
            return

        mlflow.set_tag("eval_status", "success")

        # Static metrics
        static = result.get("static_eval", {})
        _log_metric  = lambda k, v: mlflow.log_metric(k, v) if v is not None else None
        _log_param   = lambda k, v: mlflow.log_param(k, v)  if v is not None else None
        _log_tag     = lambda k, v: mlflow.set_tag(k, str(v)[:500]) if v is not None else None

        _log_metric("latency_ms",              static.get("latency_ms"))
        _log_metric("latency_score",           static.get("latency_score"))
        _log_metric("total_tokens",            static.get("total_tokens") or None)
        _log_metric("input_tokens",            static.get("input_tokens") or None)
        _log_metric("output_tokens",           static.get("output_tokens") or None)
        _log_metric("efficiency_score",        static.get("efficiency_score"))
        _log_metric("verbosity_ratio",         static.get("verbosity_ratio"))
        _log_metric("response_word_count",     static.get("response_word_count"))
        _log_metric("avg_words_per_sentence",  static.get("avg_words_per_sentence"))
        _log_tag("latency_label",              static.get("latency_label"))
        _log_tag("latency_passed",             static.get("latency_passed"))
        _log_tag("efficiency_label",           static.get("efficiency_label"))
        _log_tag("efficiency_passed",          static.get("efficiency_passed"))

        # --- DeepEval metrics (score 0-10) ---
        for metric_name, metric_data in result.get("llm_eval", {}).items():
            score  = metric_data.get("score")
            passed = metric_data.get("passed")
            _log_metric(f"deepeval_{metric_name}_score",  score)
            _log_tag(f"deepeval_{metric_name}_passed",    passed)
            _log_tag(f"deepeval_{metric_name}_reason",    metric_data.get("reason"))
            _log_tag(f"deepeval_{metric_name}_error",     metric_data.get("error"))

        # --- Custom Gemini metrics (score 0-10) ---
        for metric_name, metric_data in result.get("custom_eval", {}).items():
            score  = metric_data.get("score")
            passed = metric_data.get("passed")
            _log_metric(f"custom_{metric_name}_score",  score)
            _log_tag(f"custom_{metric_name}_passed",    passed)
            _log_tag(f"custom_{metric_name}_reason",    metric_data.get("reason"))
            _log_tag(f"custom_{metric_name}_error",     metric_data.get("error"))

        # --- Overall (score composto 0-10, grade A-F, health green/yellow/red) ---
        overall = result.get("overall", {})
        _log_metric("overall_score",  overall.get("score"))
        _log_tag("overall_grade",     overall.get("grade"))
        _log_tag("overall_health",    overall.get("health"))
        _log_tag("overall_passed",    overall.get("passed"))

        # --- Artefatos JSON completos ---
        mlflow.log_dict(result, "eval_result.json")
        mlflow.log_dict(
            {
                "user_prompt":    attrs.get("agent_tracer.agent.input", ""),
                "agent_response": attrs.get("agent_tracer.agent.output", ""),
                "tool_output":    attrs.get("agent_tracer.tool.output", ""),
            },
            "trace_inputs.json",
        )


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "Motor de Avaliação Agêntica"}


@app.post("/ingest", response_model=IngestResponse, tags=["Pipeline"])
async def ingest(payload: IngestPayload, background_tasks: BackgroundTasks):
    """
    Receives an agent trace, triggers the evaluation pipeline in background
    and registers the results in MLflow via DagsHub.
    """
    trace_id = payload.trace_id or str(uuid.uuid4())
    run_name = payload.run_name or f"trace-{trace_id[:8]}-{int(time.time())}"
    tags = payload.tags or {}

    background_tasks.add_task(
        _run_eval_pipeline,
        trace_id=trace_id,
        payload_dict=payload.attributes.model_dump(),
        run_name=run_name,
        extra_tags=tags,
    )

    return IngestResponse(
        trace_id=trace_id,
        status="accepted",
        message=f"Trace '{trace_id}' enfileirado para avaliação. Acompanhe em /dashboard.",
    )


@app.get("/dashboard", response_model=DashboardResponse, tags=["Dashboard"])
def dashboard(limit: int = 50, only_successful: bool = False):
    """
    Returns the results of all evaluations registered in MLflow/DagsHub.
    """
    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            raise HTTPException(status_code=404, detail="Experimento não encontrado no MLflow.")

        filter_string = "tags.eval_status = 'success'" if only_successful else ""

        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            max_results=limit,
            order_by=["start_time DESC"],
        )

        runs: List[EvalResult] = []
        tracking_uri = mlflow.get_tracking_uri()

        for _, row in runs_df.iterrows():
            metrics = {
                k.replace("metrics.", ""): v
                for k, v in row.items()
                if k.startswith("metrics.") and not (isinstance(v, float) and v != v)
            }
            params = {
                k.replace("params.", ""): v
                for k, v in row.items()
                if k.startswith("params.") and v is not None
            }
            tags_out = {
                k.replace("tags.", ""): str(v)
                for k, v in row.items()
                if k.startswith("tags.") and v is not None
            }

            run_url = f"{tracking_uri}/#/experiments/{experiment.experiment_id}/runs/{row['run_id']}"

            runs.append(
                EvalResult(
                    trace_id=tags_out.get("trace_id", ""),
                    run_id=row["run_id"],
                    run_name=row.get("tags.mlflow.runName", row["run_id"]),
                    status=tags_out.get("eval_status", "unknown"),
                    start_time=str(row.get("start_time", "")),
                    metrics=metrics,
                    params=params,
                    tags=tags_out,
                    mlflow_url=run_url,
                )
            )

        return DashboardResponse(
            experiment_name=EXPERIMENT_NAME,
            experiment_id=experiment.experiment_id,
            total_runs=len(runs),
            runs=runs,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao consultar MLflow: {str(e)}")


@app.get("/dashboard/{trace_id}", response_model=EvalResult, tags=["Dashboard"])
def get_trace_result(trace_id: str):
    """
    Returns the result of the evaluation of a specific trace.
    """
    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            raise HTTPException(status_code=404, detail="Experimento não encontrado no MLflow.")

        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.trace_id = '{trace_id}'",
            max_results=1,
        )

        if runs_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Trace '{trace_id}' não encontrado. Pode ainda estar sendo processado.",
            )

        row = runs_df.iloc[0]
        tracking_uri = mlflow.get_tracking_uri()

        metrics = {
            k.replace("metrics.", ""): v
            for k, v in row.items()
            if k.startswith("metrics.") and not (isinstance(v, float) and v != v)
        }
        params = {
            k.replace("params.", ""): v
            for k, v in row.items()
            if k.startswith("params.") and v is not None
        }
        tags_out = {
            k.replace("tags.", ""): str(v)
            for k, v in row.items()
            if k.startswith("tags.") and v is not None
        }

        run_url = f"{tracking_uri}/#/experiments/{experiment.experiment_id}/runs/{row['run_id']}"

        return EvalResult(
            trace_id=trace_id,
            run_id=row["run_id"],
            run_name=row.get("tags.mlflow.runName", row["run_id"]),
            status=tags_out.get("eval_status", "unknown"),
            start_time=str(row.get("start_time", "")),
            metrics=metrics,
            params=params,
            tags=tags_out,
            mlflow_url=run_url,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao consultar MLflow: {str(e)}")