"""Send a real image, then the benchmark's JSON schema, to an LMDeploy server."""
import argparse
import base64
import json
import os
from pathlib import Path
import runpy
import sys
import time
import urllib.request


def main():
    root = Path(__file__).resolve().parents[2]
    bench = root / "third_party/EmbodiedBench"
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:23333/v1")
    parser.add_argument("--image", type=Path, default=bench / "embodiedbench/evaluator/config/visual_icl_examples/eb_navigation/example0_scene0.png")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--benchmark-client", action="store_true",
                        help="Also call the repository's RemoteModel (run in embench).")
    args = parser.parse_args()
    # Local requests must not be sent through a user's external HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(args.url + "/models", timeout=30) as response:
        models = json.load(response)
    model = "Qwen/Qwen2-VL-7B-Instruct"
    assert model in [item["id"] for item in models["data"]], models
    data_url = "data:image/png;base64," + base64.b64encode(args.image.read_bytes()).decode()
    schema = runpy.run_path(str(bench / "embodiedbench/planner/planner_config/generation_guide.py"))["vlm_generation_guide"]
    records = []
    for structured in (False, True):
        prompt = "Describe the visible room and objects in this image in two sentences."
        if structured:
            prompt = ("Inspect the image and produce a JSON plan to observe the room. "
                      "The only available action is action_id 0, action_name look. "
                      "Describe the visible scene, explain your choice, and include that action.")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
            "temperature": 0,
            "max_tokens": 512,
        }
        if structured:
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "embodied_planning", "schema": schema,
            }}
        started = time.perf_counter()
        request = urllib.request.Request(
            args.url + "/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with opener.open(request, timeout=600) as response:
            result = json.load(response)
        content = result["choices"][0]["message"]["content"]
        assert content and content.strip(), result
        assert result["choices"][0]["finish_reason"] == "stop", result
        if structured:
            import jsonschema
            jsonschema.validate(json.loads(content), schema)
        record = {"structured": structured, "seconds": time.perf_counter() - started,
                  "image": str(args.image), "result": result}
        records.append(record)
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    if args.benchmark_client:
        os.environ["remote_url"] = args.url
        os.environ["OPENAI_API_KEY"] = "EMPTY"
        os.environ["NO_PROXY"] = "127.0.0.1,localhost"
        os.environ["no_proxy"] = os.environ["NO_PROXY"]
        sys.path.insert(0, str(bench))
        from embodiedbench.planner.remote_model import RemoteModel
        client = RemoteModel(model, model_type="remote")
        started = time.perf_counter()
        content = client.respond(payload["messages"])
        decoded = json.loads(content)
        jsonschema.validate(decoded, schema)
        record = {"benchmark_client": True, "seconds": time.perf_counter() - started,
                  "response": decoded}
        records.append(record)
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    print("PASS: image inference and EmbodiedBench JSON schema requests")
    if args.benchmark_client:
        print("PASS: unmodified EmbodiedBench RemoteModel client")


if __name__ == "__main__":
    main()
