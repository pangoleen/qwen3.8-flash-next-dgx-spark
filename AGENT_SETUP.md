# Agent setup prompt

Paste everything below this line into a coding agent that has shell access on
the DGX Spark. It assumes this repository is cloned in the current directory.

---

You are setting up the Qwen3.8-Flash-Next serving recipe in this repository on
an NVIDIA DGX Spark (GB10, 128 GB unified memory, aarch64). Work through the
steps in order, verify each one before moving on, and report what you found at
every step. Rules: never print, log, or paste any API key; never run this
alongside another large model server, the two do not share 128 GB; every step
that touches the Hugging Face cache should log what it wrote and why; if a
step fails, stop and report rather than improvising around it.

1. Check the box: `nvidia-smi` to confirm the GPU is reachable, `df -h $HOME`
   for at least 140 GB free (the checkpoint is ~127 GB as published), `free -g`
   and `docker ps` to confirm no other large model server is running.
2. Read this repository's README.md fully, then `serve.sh`.
3. Clone the upstream engine repo alongside this one:
   `git clone https://github.com/blazux/qwen3.8-Flash-DGX.git` and build its
   image: `cd qwen3.8-Flash-DGX && docker build -t qwen38-flash-dgx .`. Report
   the resulting image ID.
4. Copy this repository's `serve.sh` and `scripts/weight_utils.nogds.py` into
   the `qwen3.8-Flash-DGX` clone, overwriting its stock `scripts/serve.sh`.
5. Weights: follow this repository's README "Weights" section. A checkpoint
   download can take a long time at ~127 GB; if it is already present, confirm
   the snapshot directory holds `.safetensors` files before moving on, the same
   check `serve.sh` makes at launch.
6. Prepare the hybrid checkpoint (one-time, ~10 minutes, needs no other model
   server running): `MODE=hybrid` needs `scripts/prepare-hybrid.sh` run first,
   from inside the `qwen3.8-Flash-DGX` clone. Confirm it reports success before
   launching with `MODE=hybrid`.
7. Launch with the recipe this README recommends:
   `MODE=hybrid MTP=3 GPU_MEM=0.80 PORT=18300 ./serve.sh` from inside the
   `qwen3.8-Flash-DGX` clone (with this repo's `serve.sh` in place). Boot is
   8-13 minutes; wait for `docker logs -f qwen38-flash` to print "Application
   startup complete".
8. First token: `curl -s http://localhost:18300/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen3.8-flash-next","max_tokens":64,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"Write a haiku about GPUs."}]}'`.
   Report the completion.
9. One rung of the sweep: from this repository, `source .env.sample` with
   `SPARK_BASE_URL` pointed at the running server, then
   `python3 bench/ctxsweep.py --lengths 512,8192 --reps 2 --out-tokens 512`.
   Report the table. Expected generation is roughly 40-46 tok/s at both rungs;
   if it is well under that, report the full server startup log rather than
   re-running blind — the two most common causes are `MODE=nvfp4` (no hybrid
   side layers) and `SEQS` left too low under concurrent load.
10. Record the baseline: image ID, boot time, whether `FAST_LOAD` was used and
    whether it OOM-killed (see README, Traps, before turning it on), and the
    two sweep rows. Stop here if the stock recipe is all that is wanted.

## Optional: build the tuned image (+22% decode, +29% with thinking on)

Only after step 10 has a working baseline and its numbers recorded — you want a
before to compare against.

11. Read `build/README.md`. It explains the four overlays in plain terms and
    what each one costs to build.
12. From this repository, run `./build/build.sh`. It layers four overlays on
    the `qwen38-flash-dgx` image from step 3 and tags the result
    `qwen38-flash-tuned`. Expect roughly 20 minutes; overlay 02 compiles a CUDA
    kernel for `sm_121a`, and overlay 01 runs 19 unit tests as a build gate.
    Report the final image ID.
    - If a patch step fails with a SHA256 mismatch, **stop and report it**. It
      means the base image's vLLM differs from the one these overlays were
      measured against. Do not edit the expected hash to force it through: the
      published numbers would no longer apply to what you built. `build/README.md`
      explains how to re-anchor deliberately.
13. Stop the running server (`./stop.sh`), then launch the tuned image:
    `IMAGE=qwen38-flash-tuned MODE=hybrid MTP=3 GPU_MEM=0.80 PORT=18300 ./serve.sh`.
    Only one of these servers can run at a time.
14. Re-run the same two sweep rungs from step 9 against the tuned image and put
    the two tables side by side. Expect roughly +20% or better on generation at
    both rungs. If it is not faster, report the startup log rather than
    re-running: the usual cause is the overlays being present but not switched
    on, so check that the launch enabled `VLLM_PLE_STAGED`, the draft
    vocabulary, and `use_local_argmax_reduction`, per `build/README.md`.
15. If prefix caching matters for your workload, run
    `python3 bench/suffix_gate.py "$SPARK_BASE_URL" "$SPARK_MODEL" 60000 15000 8`.
    It grows a shared prefix turn by turn and checks every answer. Report the
    pass/fail line and the median turn time.
16. Finish with a summary: both image IDs, both sweep tables, the measured
    gain, and the gate result if you ran it.
