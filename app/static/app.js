(() => {
  const state = {
    file: null,
    meta: null,
    mode: "detect",
    objectUrl: null,
  };

  const $ = (id) => document.getElementById(id);

  function setStatus(el, message, kind) {
    el.textContent = message || "";
    el.classList.toggle("ok", kind === "ok");
    el.classList.toggle("err", kind === "err");
  }

  function pretty(obj) {
    return JSON.stringify(obj, null, 2);
  }

  function setStage(src, label) {
    const frame = document.querySelector(".stage-frame");
    const img = $("stageImg");
    if (!src) {
      frame.classList.remove("has-image");
      img.removeAttribute("src");
      $("stageEmpty").textContent = "Pick a sample or upload an image";
      return;
    }
    img.src = src;
    frame.classList.add("has-image");
    if (label) $("fileLabel").textContent = label;
  }

  function setImageFile(file, previewUrl) {
    state.file = file;
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = previewUrl || URL.createObjectURL(file);
    setStage(state.objectUrl, file.name);
    document.querySelectorAll(".sample-btn").forEach((btn) => {
      btn.setAttribute("aria-selected", String(btn.dataset.file === file.name));
    });
  }

  async function fetchMeta() {
    const res = await fetch("/demo/meta");
    if (!res.ok) throw new Error(`meta ${res.status}`);
    return res.json();
  }

  function renderMeta(meta) {
    state.meta = meta;
    $("subtitle").textContent = meta.subtitle;
    $("metrics").innerHTML = [
      ["imgsz", meta.imgsz],
      ["detect", meta.detect_display_confidence_default],
      ["evidence", meta.ask_evidence_confidence],
      ["answer", meta.ask_answer_confidence],
      ["model", meta.model_ready ? "ready" : "down"],
    ]
      .map(([k, v]) => `<span class="chip"><b>${k}</b> ${v}</span>`)
      .join("");

    const links = meta.links || {};
    $("links").innerHTML = [
      ["API", links.api_docs],
      ["Repo", links.repo],
      ["Weights", links.weights],
      ["Memo", links.memo],
    ]
      .filter(([, href]) => href)
      .map(([label, href]) => `<a href="${href}" ${href.startsWith("http") ? 'target="_blank" rel="noreferrer"' : ""}>${label}</a>`)
      .join("");

    const samples = $("samples");
    samples.innerHTML = "";
    for (const s of meta.samples || []) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "sample-btn";
      btn.dataset.file = s.file;
      btn.setAttribute("role", "option");
      btn.setAttribute("aria-selected", "false");
      btn.textContent = s.label;
      btn.title = s.file;
      btn.addEventListener("click", () => loadSample(s.file));
      samples.appendChild(btn);
    }

    const prompt = $("prompt");
    prompt.innerHTML = "";
    for (const p of meta.prompts || []) {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = p.label;
      opt.dataset.text = p.text;
      prompt.appendChild(opt);
    }
    if (meta.prompts?.length) {
      $("question").value = meta.prompts[0].text;
    }

    if (meta.model_error) {
      setStatus($("detectStatus"), meta.model_error, "err");
    }
  }

  async function loadSample(name) {
    const res = await fetch(`/demo/samples/${name}`);
    if (!res.ok) throw new Error(`sample ${res.status}`);
    const blob = await res.blob();
    const file = new File([blob], name, { type: blob.type || "image/jpeg" });
    setImageFile(file);
  }

  function onFileChange(ev) {
    const file = ev.target.files?.[0];
    if (!file) return;
    setImageFile(file);
  }

  function pastePrompt() {
    const opt = $("prompt").selectedOptions[0];
    if (!opt) return;
    $("question").value = opt.dataset.text || "";
  }

  function switchTab(which) {
    state.mode = which;
    const detect = which === "detect";
    $("tabDetect").setAttribute("aria-selected", String(detect));
    $("tabAsk").setAttribute("aria-selected", String(!detect));
    $("detectControls").classList.toggle("hidden", !detect);
    $("askControls").classList.toggle("hidden", detect);
    $("stageLabel").textContent = detect ? "Image / detections" : "Image / evidence";
  }

  async function runDetect() {
    if (!state.file) {
      setStatus($("detectStatus"), "Select a sample or upload first.", "err");
      return;
    }
    const btn = $("runDetect");
    btn.disabled = true;
    setStatus($("detectStatus"), "Running…", null);
    try {
      const body = new FormData();
      body.append("image", state.file, state.file.name);
      body.append("confidence", $("conf").value);
      const res = await fetch("/demo/detect", { method: "POST", body });
      const data = await res.json();
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : `HTTP ${res.status}`);
      if (data.annotated_image) setStage(data.annotated_image, state.file.name);
      $("userJson").textContent = pretty(data.user);
      $("backendJson").textContent = pretty(data.backend);
      const ms = data.backend?.timing_ms?.detect;
      setStatus($("detectStatus"), ms != null ? `${ms} ms · ${data.user.count} shown` : "Done", "ok");
    } catch (err) {
      setStatus($("detectStatus"), String(err.message || err), "err");
    } finally {
      btn.disabled = false;
    }
  }

  async function runAsk() {
    const question = ($("question").value || "").trim();
    if (!question) {
      setStatus($("askStatus"), "Enter a question.", "err");
      return;
    }
    const btn = $("runAsk");
    btn.disabled = true;
    setStatus($("askStatus"), "Running…", null);
    try {
      const body = new FormData();
      body.append("question", question);
      if (state.file) body.append("image", state.file, state.file.name);
      const res = await fetch("/demo/ask", { method: "POST", body });
      const data = await res.json();
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : `HTTP ${res.status}`);
      $("answerText").textContent = data.answer_text || "";
      if (data.annotated_image) setStage(data.annotated_image, state.file?.name || "evidence");
      $("userJson").textContent = pretty(data.user);
      $("backendJson").textContent = pretty(data.backend);
      const t = data.backend?.timing_ms || {};
      setStatus(
        $("askStatus"),
        `${data.backend?.route_precheck || "?"} · ${t.detect ?? "—"} / ${t.reason ?? "—"} ms`,
        "ok"
      );
    } catch (err) {
      setStatus($("askStatus"), String(err.message || err), "err");
    } finally {
      btn.disabled = false;
    }
  }

  function wireCopy() {
    document.querySelectorAll(".copy").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const pre = $(btn.dataset.copy);
        await navigator.clipboard.writeText(pre.textContent || "");
        const prev = btn.textContent;
        btn.textContent = "Copied";
        setTimeout(() => {
          btn.textContent = prev;
        }, 900);
      });
    });
  }

  function wire() {
    $("conf").addEventListener("input", () => {
      $("confOut").textContent = Number($("conf").value).toFixed(2);
    });
    $("file").addEventListener("change", onFileChange);
    $("prompt").addEventListener("change", pastePrompt);
    $("tabDetect").addEventListener("click", () => switchTab("detect"));
    $("tabAsk").addEventListener("click", () => switchTab("ask"));
    $("runDetect").addEventListener("click", runDetect);
    $("runAsk").addEventListener("click", runAsk);
    wireCopy();
  }

  wire();
  fetchMeta()
    .then(async (meta) => {
      renderMeta(meta);
      const first = meta.samples?.[0]?.file;
      if (first) await loadSample(first);
    })
    .catch((err) => {
      $("subtitle").textContent = `Failed to load /demo/meta: ${err}`;
    });
})();
