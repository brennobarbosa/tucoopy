(function () {
  let installed = false;
  let lastRenderKey = null;

  // Config do triângulo (coordenadas no viewBox)
  const V = 100; // v(N)
  const A = { x: 300, y: 60  };  // P1=100
  const B = { x: 100, y: 420 };  // P2=100
  const C = { x: 500, y: 420 };  // P3=100

  function ensureTooltipEl() {
    let tip = document.getElementById("cgTip");
    if (!tip) {
      tip = document.createElement("div");
      tip.id = "cgTip";
      tip.className = "cg-tooltip";
      document.body.appendChild(tip);
    }
    return tip;
  }

  function barycentric(P, A, B, C) {
    const det =
      (B.y - C.y) * (A.x - C.x) +
      (C.x - B.x) * (A.y - C.y);

    const l1 =
      ((B.y - C.y) * (P.x - C.x) +
       (C.x - B.x) * (P.y - C.y)) / det;

    const l2 =
      ((C.y - A.y) * (P.x - C.x) +
       (A.x - C.x) * (P.y - C.y)) / det;

    const l3 = 1 - l1 - l2;

    return [l1, l2, l3];
  }

  function insideTriangle(l1, l2, l3, eps = 1e-6) {
    return l1 >= -eps && l2 >= -eps && l3 >= -eps;
  }

  // Converte coordenada do mouse (client) para coordenada do SVG (viewBox)
  function clientToSvgPoint(svg, clientX, clientY) {
    const pt = svg.createSVGPoint();
    pt.x = clientX;
    pt.y = clientY;
    const m = svg.getScreenCTM();
    if (!m) return null;
    const inv = m.inverse();
    return pt.matrixTransform(inv);
  }

  function install() {
    if (installed) return;
    installed = true;

    document.addEventListener("mousemove", (e) => {
      const tip = ensureTooltipEl();

      const target = e.target.closest?.(".cg-svg .region");
      if (!target) {
        tip.style.display = "none";
        lastRenderKey = null;
        return;
      }

      // Tooltip próximo ao mouse (viewport)
      tip.style.display = "block";
      tip.style.position = "fixed";
      tip.style.left = (e.clientX + 14) + "px";
      tip.style.top  = (e.clientY + 14) + "px";

      const svg = target.ownerSVGElement;
      const p = svg ? clientToSvgPoint(svg, e.clientX, e.clientY) : null;

      // Texto base do elemento (opcional)
      const baseTex = target.getAttribute("data-tex") || "";

      // Se conseguimos calcular (x1,x2,x3) e o ponto está dentro do triângulo
      let allocTex = "";
      if (p) {
        const [l1, l2, l3] = barycentric(p, A, B, C);
        if (insideTriangle(l1, l2, l3)) {
          const x1 = V * l1;
          const x2 = V * l2;
          const x3 = V * l3;

          // evita "-0.0"
          const f = (z) => (Math.abs(z) < 0.05 ? 0 : z);

          allocTex =
            `\\\\\\\\\\small (x_1,x_2,x_3)=(${f(x1).toFixed(1)},\\,${f(x2).toFixed(1)},\\,${f(x3).toFixed(1)})`;
        }
      }

      // Conteúdo final do tooltip
      const tex = baseTex + allocTex;

      // Só renderiza se mudou (conteúdo + alvo)
      const renderKey = (target.id || "") + "::" + tex;
      if (renderKey === lastRenderKey) return;
      lastRenderKey = renderKey;

      tip.innerHTML = "";

      if (typeof katex !== "undefined") {
        katex.render(tex || "\\text{ }", tip, {
          throwOnError: false,
          displayMode: false
        });
      } else {
        tip.textContent = tex;
      }
    });
  }

  // MkDocs Material SPA support
  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(() => {
      install();
      lastRenderKey = null;
      const tip = document.getElementById("cgTip");
      if (tip) tip.style.display = "none";
    });
  } else {
    document.addEventListener("DOMContentLoaded", install);
  }
})();
