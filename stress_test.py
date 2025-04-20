import asyncio
import httpx
import time
import numpy as np

async def measure_inference(prompt: str) -> dict:
    start_time = time.time()
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post("http://localhost:11434/api/generate", json={
            "model": "qwen2.5-coder",
            "prompt": prompt,
            "stream": False
        })
    end_time = time.time()
    
    data = resp.json()
    duration = end_time - start_time
    tps = data.get('eval_count', 0) / (data.get('eval_duration', 1) / 1e9)
    return {"tps": tps, "total_time": duration}

async def run_suite():
    prompts = ["Write a python script to reverse a string."] * 5
    print(f"Firing {len(prompts)} concurrent requests to local Qwen 2.5...")
    results = await asyncio.gather(*(measure_inference(p) for p in prompts))
    
    avg_tps = np.mean([r['tps'] for r in results])
    print(f"Average Throughput: {avg_tps:.2f} Tokens/Second")

if __name__ == "__main__":
    # asyncio.run(run_suite())
    print("Benchmark suite initialized.")
