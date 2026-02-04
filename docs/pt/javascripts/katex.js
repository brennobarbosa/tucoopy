// KaTeX rendering for MkDocs Material + pymdownx.arithmatex.
//
// We use arithmatex with generic=true, which wraps TeX into elements with the
// `arithmatex` class. Here we render those elements with KaTeX.
//
// Docs: https://squidfunk.github.io/mkdocs-material/reference/math/

function renderArithmatexElement(el) {
  if (typeof katex === "undefined") return;

  const raw = (el.textContent || "").trim();
  if (!raw) return;

  // arithmatex generic wraps inline as \( ... \) and display as \[ ... \]
  let displayMode = false;
  let tex = raw;

  if (tex.startsWith("\\[") && tex.endsWith("\\]")) {
    displayMode = true;
    tex = tex.slice(2, -2).trim();
  } else if (tex.startsWith("\\(") && tex.endsWith("\\)")) {
    displayMode = false;
    tex = tex.slice(2, -2).trim();
  } else if (tex.startsWith("$$") && tex.endsWith("$$")) {
    displayMode = true;
    tex = tex.slice(2, -2).trim();
  } else if (tex.startsWith("$") && tex.endsWith("$")) {
    displayMode = false;
    tex = tex.slice(1, -1).trim();
  }

  try {
    katex.render(tex, el, {
      displayMode,
      throwOnError: false,
      strict: "warn",
    });
  } catch (e) {
    // Fall back to raw text if KaTeX fails.
  }
}

document$.subscribe(function () {
  document.querySelectorAll(".arithmatex").forEach(renderArithmatexElement);
});
