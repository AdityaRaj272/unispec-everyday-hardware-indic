"""
Builds the final report PDF from results/ and figures/.

    python report/build_report.py            # -> report/report.pdf
    python report/build_report.py --plots    # regenerate figures first

Every table and most numbers are computed from the result files (see
report_data.py), so after adding a device's results folder a rebuild updates
the report. Sentences about one particular device are generated per run;
interpretive paragraphs that compare device classes may need a manual read
after adding a very different kind of device.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib import mathtext  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.lib.utils import ImageReader  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether,  # noqa: E402
                                NextPageTemplate, PageBreak, PageTemplate, Paragraph, Preformatted, Spacer, Table,
                                TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents  # noqa: E402

import report_data as rd  # noqa: E402

ROOT = rd.ROOT
TITLE = "Testing the Limits of Training-Free Speculative Decoding: Everyday Hardware and Indian Languages"
SHORT_TITLE = "Training-free speculative decoding on everyday hardware"
AUTHORS = ["Pranjal Kirar (230106050)", "Aditya Raj (230121004)", "Khushal Nagwal (230106035)",
           "Babita Verma (230106015)", "Yash Pratap Singh (230106084)"]
COURSE = "DA4022 Hardware-Aware Deep Learning · Final project report"
REPO = "https://github.com/AdityaRaj272/unispec-everyday-hardware-indic"

INK = colors.HexColor("#111111")
GREY = colors.HexColor("#555555")
RULE = colors.HexColor("#999999")
SHADE = colors.HexColor("#f1f3f6")

# ---------------------------------------------------------------------------
# Fonts and styles (fonts ship with matplotlib, so the build is the same on every OS)
# ---------------------------------------------------------------------------

_TTF = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
for name, file in [("Serif", "STIXGeneral.ttf"), ("Serif-Bold", "STIXGeneralBol.ttf"),
                   ("Serif-Italic", "STIXGeneralItalic.ttf"), ("Serif-BoldItalic", "STIXGeneralBolIta.ttf"),
                   ("Mono", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(name, os.path.join(_TTF, file)))
pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-Bold", italic="Serif-Italic",
                              boldItalic="Serif-BoldItalic")

S = {
    "title": ParagraphStyle("title", fontName="Serif-Bold", fontSize=17, leading=21, alignment=TA_CENTER,
                            spaceAfter=10, textColor=INK),
    "authors": ParagraphStyle("authors", fontName="Serif", fontSize=10.5, leading=14, alignment=TA_CENTER),
    "meta": ParagraphStyle("meta", fontName="Serif-Italic", fontSize=9.5, leading=13, alignment=TA_CENTER,
                           textColor=GREY),
    "abs_head": ParagraphStyle("abs_head", fontName="Serif-Bold", fontSize=10.5, leading=14, alignment=TA_CENTER,
                               spaceBefore=12, spaceAfter=4),
    "abstract": ParagraphStyle("abstract", fontName="Serif", fontSize=9.8, leading=13.2, alignment=TA_JUSTIFY,
                               leftIndent=1.1 * cm, rightIndent=1.1 * cm),
    "h1": ParagraphStyle("h1", fontName="Serif-Bold", fontSize=13, leading=16, spaceBefore=14, spaceAfter=6,
                         keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="Serif-Bold", fontSize=11, leading=14, spaceBefore=9, spaceAfter=4,
                         keepWithNext=1),
    "body": ParagraphStyle("body", fontName="Serif", fontSize=10.5, leading=14.2, alignment=TA_JUSTIFY,
                           spaceAfter=6),
    "bullet": ParagraphStyle("bullet", fontName="Serif", fontSize=10.5, leading=14.2, alignment=TA_JUSTIFY,
                             leftIndent=14, bulletIndent=3, spaceAfter=3),
    "caption": ParagraphStyle("caption", fontName="Serif", fontSize=9, leading=11.8, alignment=TA_JUSTIFY,
                              textColor=INK, spaceBefore=3, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="Serif", fontSize=8.8, leading=10.8, alignment=TA_LEFT),
    "cellb": ParagraphStyle("cellb", fontName="Serif-Bold", fontSize=8.8, leading=10.8, alignment=TA_LEFT),
    "ref": ParagraphStyle("ref", fontName="Serif", fontSize=9.4, leading=12.4, leftIndent=18, firstLineIndent=-18,
                          spaceAfter=3),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=8, leading=10.4),
    "toc1": ParagraphStyle("toc1", fontName="Serif", fontSize=10, leading=13, leftIndent=0),
    "toc2": ParagraphStyle("toc2", fontName="Serif", fontSize=9.5, leading=12, leftIndent=16, textColor=GREY),
}

W = A4[0] - 2 * 2.3 * cm          # text width


def x(v: float) -> str:
    return f"{v:.2f}×"


def pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# Document scaffolding: numbered headings, figures, tables, equations
# ---------------------------------------------------------------------------

class Report:
    def __init__(self):
        self.story = []
        self.h = [0, 0]
        self.fig = 0
        self.tab = 0
        self.eq = 0
        self.refs = {}

    # text
    def p(self, text: str, style: str = "body"):
        self.story.append(Paragraph(text, S[style]))

    def bullets(self, items):
        for it in items:
            self.story.append(Paragraph(it, S["bullet"], bulletText="•"))
        self.story.append(Spacer(1, 3))

    def numbered(self, items):
        for i, it in enumerate(items, 1):
            self.story.append(Paragraph(it, S["bullet"], bulletText=f"{i}."))
        self.story.append(Spacer(1, 3))

    def h1(self, text: str, numbered: bool = True):
        if numbered:
            self.h = [self.h[0] + 1, 0]
            text = f"{self.h[0]}&nbsp;&nbsp;{text}"
        self.story.append(CondPageBreak(3 * cm))
        para = Paragraph(text, S["h1"])
        para._toc = (0, text.replace("&nbsp;", " "))
        self.story.append(para)

    def h2(self, text: str):
        self.h[1] += 1
        text = f"{self.h[0]}.{self.h[1]}&nbsp;&nbsp;{text}"
        para = Paragraph(text, S["h2"])
        para._toc = (1, text.replace("&nbsp;", " "))
        self.story.append(para)

    def appendix(self, letter: str, text: str):
        self.story.append(CondPageBreak(3 * cm))
        t = f"Appendix {letter}&nbsp;&nbsp;{text}"
        para = Paragraph(t, S["h1"])
        para._toc = (0, t.replace("&nbsp;", " "))
        self.story.append(para)

    # floats
    def figure(self, path: str, caption: str, width: float = W):
        self.fig += 1
        img = ImageReader(path)
        iw, ih = img.getSize()
        h = width * ih / iw
        self.story.append(KeepTogether([
            Spacer(1, 4), Image(path, width=width, height=h),
            Paragraph(f"<b>Figure {self.fig}.</b> {caption}", S["caption"])]))
        return self.fig

    def table(self, header, rows, caption: str, widths=None, bold_first_col=False, align_right_from=None,
              notes: str = ""):
        self.tab += 1
        data = [[Paragraph(f"<b>{h}</b>", S["cell"]) for h in header]]
        for r in rows:
            data.append([Paragraph(str(c), S["cellb"] if (bold_first_col and j == 0) else S["cell"])
                         for j, c in enumerate(r)])
        t = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
        style = [("LINEABOVE", (0, 0), (-1, 0), 0.9, INK), ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK),
                 ("LINEBELOW", (0, -1), (-1, -1), 0.9, INK), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                 ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                 ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
        t.setStyle(TableStyle(style))
        cap = Paragraph(f"<b>Table {self.tab}.</b> {caption}", S["caption"])
        parts = [cap, t]
        if notes:
            parts.append(Paragraph(notes, ParagraphStyle("n", parent=S["caption"], fontSize=8.3, leading=10.5,
                                                        spaceBefore=3)))
        parts.append(Spacer(1, 8))
        self.story.append(KeepTogether(parts))
        return self.tab

    def equation(self, tex: str, size: float = 13.5):
        """Display equation rendered with matplotlib's mathtext (STIX fonts), numbered on the right."""
        self.eq += 1
        buf = io.BytesIO()
        mathtext.math_to_image(f"${tex}$", buf, dpi=300, format="png",
                               prop=FontProperties(size=size, math_fontfamily="stix"))
        buf.seek(0)
        iw, ih = ImageReader(buf).getSize()
        buf.seek(0)
        img = Image(buf, width=iw * 72 / 300, height=ih * 72 / 300)
        t = Table([["", img, Paragraph(f"({self.eq})", S["body"])]], colWidths=[W * 0.1, W * 0.8, W * 0.1])
        t.setStyle(TableStyle([("ALIGN", (1, 0), (1, 0), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("ALIGN", (2, 0), (2, 0), "RIGHT")]))
        self.story.append(t)
        return self.eq

    def code(self, text: str):
        pre = Preformatted(text, S["code"])
        t = Table([[pre]], colWidths=[W])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), SHADE), ("BOX", (0, 0), (-1, -1), 0.4, RULE),
                               ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        self.story.append(KeepTogether([t, Spacer(1, 8)]))


class Doc(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(path, pagesize=A4, leftMargin=2.3 * cm, rightMargin=2.3 * cm, topMargin=2.2 * cm,
                         bottomMargin=2.2 * cm, title=TITLE, author=", ".join(a.split(" (")[0] for a in AUTHORS),
                         subject=COURSE)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate("first", [frame], onPage=self._first),
                               PageTemplate("rest", [frame], onPage=self._rest)])

    def _first(self, c, d):
        c.saveState()
        c.setFont("Serif", 9)
        c.setFillColor(GREY)
        c.drawCentredString(A4[0] / 2, 1.2 * cm, str(d.page))
        c.restoreState()

    def _rest(self, c, d):
        c.saveState()
        c.setFont("Serif-Italic", 8.5)
        c.setFillColor(GREY)
        c.drawString(self.leftMargin, A4[1] - 1.4 * cm, SHORT_TITLE)
        c.drawRightString(A4[0] - self.rightMargin, A4[1] - 1.4 * cm, "DA4022 final report")
        c.setStrokeColor(RULE)
        c.setLineWidth(0.4)
        c.line(self.leftMargin, A4[1] - 1.55 * cm, A4[0] - self.rightMargin, A4[1] - 1.55 * cm)
        c.setFont("Serif", 9)
        c.drawCentredString(A4[0] / 2, 1.2 * cm, str(d.page))
        c.restoreState()

    def afterFlowable(self, f):
        if hasattr(f, "_toc"):
            level, text = f._toc
            key = f"h{id(f)}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
            self.notify("TOCEntry", (level, text, self.page, key))


# ---------------------------------------------------------------------------
# References (numbered in order of this list)
# ---------------------------------------------------------------------------

REFS = [
    ("unispec", "D.-T. Do, N.-K. Le, and N. L. Minh. UniSpec: Training-free speculative decoding for robust LLM "
                "acceleration across languages and hardware. In <i>Proceedings of the 64th Annual Meeting of the "
                "Association for Computational Linguistics (ACL)</i>, pages 6288–6310, 2026. "
                "https://aclanthology.org/2026.acl-long.285/"),
    ("ppd", "H. Chen, W. Luk, K. F. C. Yiu, R. Li, K. Mishchenko, S. I. Venieris, and H. Fan. Hardware-aware "
            "parallel prompt decoding for memory-efficient acceleration of LLM inference. In <i>Findings of the "
            "Association for Computational Linguistics: EMNLP 2025</i>, pages 2221–2238, 2025. "
            "https://aclanthology.org/2025.findings-emnlp.120/"),
    ("leviathan", "Y. Leviathan, M. Kalman, and Y. Matias. Fast inference from transformers via speculative "
                  "decoding. In <i>Proceedings of the 40th International Conference on Machine Learning (ICML)</i>, 2023."),
    ("chen", "C. Chen, S. Borgeaud, G. Irving, J.-B. Lespiau, L. Sifre, and J. Jumper. Accelerating large language "
             "model decoding with speculative sampling. arXiv:2302.01318, 2023."),
    ("medusa", "T. Cai, Y. Li, Z. Geng, H. Peng, J. D. Lee, D. Chen, and T. Dao. Medusa: Simple LLM inference "
               "acceleration framework with multiple decoding heads. In <i>Proceedings of the 41st International "
               "Conference on Machine Learning (ICML)</i>, 2024."),
    ("eagle", "Y. Li, F. Wei, C. Zhang, and H. Zhang. EAGLE: Speculative sampling requires rethinking feature "
              "uncertainty. In <i>Proceedings of the 41st International Conference on Machine Learning (ICML)</i>, 2024."),
    ("lookahead", "Y. Fu, P. Bailis, I. Stoica, and H. Zhang. Break the sequential dependency of LLM inference "
                  "using lookahead decoding. In <i>Proceedings of the 41st International Conference on Machine "
                  "Learning (ICML)</i>, 2024."),
    ("pld", "A. Saxena. Prompt lookup decoding. Software, https://github.com/apoorvumang/prompt-lookup-decoding, 2023."),
    ("rest", "Z. He, Z. Zhong, T. Cai, J. D. Lee, and D. He. REST: Retrieval-based speculative decoding. In "
             "<i>Proceedings of NAACL-HLT</i>, 2024."),
    ("specinfer", "X. Miao et al. SpecInfer: Accelerating large language model serving with tree-based speculative "
                  "inference and verification. In <i>Proceedings of ASPLOS</i>, 2024."),
    ("sequoia", "Z. Chen, A. May, R. Svirschevski, Y. Huang, M. Ryabinin, Z. Jia, and B. Chen. Sequoia: Scalable, "
                "robust, and hardware-aware speculative decoding. arXiv:2402.12374, 2024."),
    ("specbench", "H. Xia, Z. Yang, Q. Dong, P. Wang, Y. Li, T. Ge, T. Liu, W. Li, and Z. Sui. Unlocking efficiency "
                  "in large language model inference: A comprehensive survey of speculative decoding. In "
                  "<i>Findings of the Association for Computational Linguistics: ACL 2024</i>, 2024."),
    ("roofline", "S. Williams, A. Waterman, and D. Patterson. Roofline: An insightful visual performance model for "
                 "multicore architectures. <i>Communications of the ACM</i>, 52(4):65–76, 2009."),
    ("vllm", "W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica. "
             "Efficient memory management for large language model serving with PagedAttention. In "
             "<i>Proceedings of the 29th Symposium on Operating Systems Principles (SOSP)</i>, 2023."),
    ("xlsum", "T. Hasan, A. Bhattacharjee, M. S. Islam, K. Mubasshir, Y.-F. Li, Y.-B. Kang, M. S. Rahman, and "
              "R. Shahriyar. XL-Sum: Large-scale multilingual abstractive summarization for 44 languages. In "
              "<i>Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021</i>, 2021."),
    ("qwen", "Qwen Team. Qwen2.5 technical report. arXiv:2412.15115, 2024."),
    ("de", "R. Storn and K. Price. Differential evolution: A simple and efficient heuristic for global optimization "
           "over continuous spaces. <i>Journal of Global Optimization</i>, 11(4):341–359, 1997."),
    ("transformers", "T. Wolf et al. Transformers: State-of-the-art natural language processing. In "
                     "<i>Proceedings of EMNLP: System Demonstrations</i>, 2020."),
    ("pytorch", "A. Paszke et al. PyTorch: An imperative style, high-performance deep learning library. In "
                "<i>Advances in Neural Information Processing Systems (NeurIPS)</i>, 2019."),
]
REF_NUM = {k: i + 1 for i, (k, _) in enumerate(REFS)}


def cite(*keys) -> str:
    return "[" + ", ".join(str(REF_NUM[k]) for k in keys) + "]"


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def build(results_root: str, figures_dir: str, out_path: str):
    runs = rd.load_runs(results_root)
    mains = rd.main_runs(runs)
    cpus = [r for r in mains if r.kind == "cpu"]
    gpus = [r for r in mains if r.kind != "cpu"]
    cpu = cpus[0]
    gpu = next((r for r in gpus if r.kind == "cuda"), gpus[0])
    ext = next((r for r in runs if r.kind == gpu.kind and r.short == gpu.short and r.tag == "wide-grid"), None)
    gpu_best = ext or gpu
    swept = [r for r in runs if r.sweep]
    fig = lambda name: os.path.join(figures_dir, name)  # noqa: E731

    # headline numbers -------------------------------------------------------
    p_cpu, p_gpu = rd.profile_at(cpu), rd.profile_at(gpu_best)
    cpu_ratio33 = p_cpu[33] / p_cpu[1]
    gpu_ratio33 = p_gpu[33] / p_gpu[1]
    gpu_maxn = max(p_gpu)
    tau_runs, common, tau_aggs, tau_gs, tau_gap = rd.tau_on_common(swept)
    cpu_bg, cpu_peak = rd.best_g(cpu)
    gpu_bg, gpu_peak = rd.best_g(gpu_best)
    cpu_g32 = cpu.sweep[32]["speedup"]
    slow32 = rd.slower_than_baseline(cpu, 32)
    wrong = rd.wrong_g(runs)
    fert = rd.fertility(runs)
    qwen_key = next(iter(fert))
    en_fert = fert[qwen_key]["english"]["tokens_per_word"]
    ind_fert = [fert[qwen_key][l]["tokens_per_word"] for l in rd.LANG_ORDER[1:]]
    lang_by_run = {r.folder: rd.language_rows(r) for r in mains if r.has("language")}
    cpu_lang = {x["lang"]: x for x in lang_by_run.get(cpu.folder, [])}
    gpu_lang = {x["lang"]: x for x in lang_by_run.get(gpu.folder, [])}
    ind = rd.LANG_ORDER[1:]
    n_dev = len(mains)
    dev_list = ", ".join(r.label for r in mains[:-1]) + (" and " if n_dev > 1 else "") + mains[-1].label

    R = Report()
    R.story.append(NextPageTemplate("rest"))

    # ---- title and abstract --------------------------------------------------
    R.p(TITLE, "title")
    R.p(" · ".join(AUTHORS[:3]) + "<br/>" + " · ".join(AUTHORS[3:]), "authors")
    R.p(f"{COURSE} · {dt.date.today():%B %Y}", "meta")
    R.p("Abstract", "abs_head")
    R.p(
        "Speculative decoding speeds up a language model by drafting several future tokens cheaply and verifying "
        "them in one forward pass, while producing exactly the text that normal decoding would. Two recent "
        f"methods, UniSpec {cite('unispec')} and Hardware-Aware Parallel Prompt Decoding (PPD) {cite('ppd')}, "
        "choose how many tokens to draft with the same hardware-aware rule: maximise tokens accepted per pass, "
        "τ(g), divided by the pass time, s(g), measured on the target device. Both evaluate this rule only on "
        "data-centre and workstation GPUs and name weak or unstable hardware and morphologically rich languages "
        f"as open gaps. We test the rule on {dev_list}, using a training-free n-gram drafter, "
        "Qwen2.5-0.5B-Instruct and news summarisation in English, Hindi, Marathi, Bengali and Tamil. Acceptance "
        f"is device-independent: on articles run on every device, τ agrees within {tau_gap:.2f}. Pass time is not: "
        f"checking 32 extra tokens costs {gpu_ratio33:.2f} times a one-token pass on the {gpu.short} but "
        f"{cpu_ratio33:.1f} times on the {cpu.short}. The best draft size is therefore {cpu_bg} on the CPU "
        f"({x(cpu_peak)}) and {gpu_bg} or more on the GPU ({x(gpu_peak)}). Using a draft size calibrated on "
        "another device loses 6–14%, and on articles that copy little from their source a large draft size is "
        "slower than normal decoding. UniSpec-style calibration and grid search pick good draft sizes but "
        "over-predict the gain, because the speedup is governed by how much the output copies its input. Qwen's "
        f"tokenizer splits the Indian languages into {min(ind_fert):.1f}–{max(ind_fert):.1f} tokens per word "
        f"against {en_fert:.2f} for English; this raises acceptance, giving up to "
        f"{x(max(gpu_lang[l]['e2e'] for l in ind))} end to end on the GPU, but makes prompts so long that reading "
        f"them limits the CPU's end-to-end gain to {x(min(cpu_lang[l]['e2e'] for l in ind))}–"
        f"{x(max(cpu_lang[l]['e2e'] for l in ind))}.", "abstract")
    R.story.append(PageBreak())
    toc = TableOfContents(levelStyles=[S["toc1"], S["toc2"]], dotsMinLevel=0)
    R.p("Contents", "abs_head")
    R.story.append(toc)
    R.story.append(PageBreak())

    # ---- 1 Introduction --------------------------------------------------------
    R.h1("Introduction")
    R.p("Large language models generate text autoregressively: one token per forward pass, and each pass needs the "
        "token chosen by the pass before it. At batch size one this is limited by memory bandwidth rather than by "
        "arithmetic, because every pass must stream all model weights from memory no matter how few tokens it "
        f"processes {cite('roofline', 'vllm')}. On a GPU, a pass over ten tokens therefore costs almost the same as "
        "a pass over one, and most of the device's arithmetic capacity is idle during generation.")
    R.p(f"Speculative decoding {cite('leviathan', 'chen')} puts that idle capacity to work. A cheap drafter "
        "proposes the next few tokens; the model scores all of them in a single pass; every proposed token up to "
        "the first disagreement is kept, followed by the model's own token at that position. Under greedy decoding "
        "the result is identical to normal decoding, so the method is lossless and only latency changes. The "
        "drafter can be a smaller model, extra trained heads or tokens, or a lookup of text already in the context "
        f"{cite('medusa', 'eagle', 'lookahead', 'pld', 'rest')}.")
    R.p("How many tokens to draft is a hardware question. More drafted tokens raise the number accepted per pass but "
        "also lengthen each pass, by an amount that depends on the device. UniSpec "
        f"{cite('unispec')} and PPD {cite('ppd')} resolve this the same way: measure the device and choose the draft "
        "size that maximises accepted tokens per unit time. Both papers evaluate only on NVIDIA data-centre and "
        "workstation GPUs, and both list weak, unstable or shared hardware and morphologically rich languages "
        "among their limitations.")
    R.p("This project tests the shared rule where the papers did not: on an ordinary laptop processor and a free "
        "cloud GPU, under sustained load, and in four Indian languages. It continues our proposal and mid-project "
        "report. At the mid-project stage, a seven-method benchmark of our decoding engine on a T4 GPU found that a "
        "1B-parameter draft model gave 1.13× while n-gram drafting gave only 1.04× on short open-domain questions, "
        "with 12% of drafted tokens accepted, because such answers contain nothing to copy. The present study "
        "therefore uses a grounded task, news summarisation, in which outputs reuse input text. Our contributions "
        "are:")
    R.bullets([
        "a portable, lossless speculative-decoding benchmark that runs unchanged on CPUs and GPUs and implements three ways of choosing the draft size (grid search, UniSpec's regression fit and "
        "PPD's measure-once split);",
        "a direct test of the assumption both papers rely on, that acceptance does not depend on the hardware;",
        "measurements of how far the optimal draft size moves between a CPU and a GPU, and what a mis-tuned draft "
        "size costs;",
        "an analysis of Indian languages that separates the effect of tokenization on acceptance from its effect on "
        "prompt length, and shows that the device decides which effect dominates.",
    ])

    # ---- 2 Background --------------------------------------------------------
    R.h1("Background and related work")
    R.h2("Autoregressive decoding is memory-bound")
    R.p("A decoder-only transformer with P parameters performs roughly 2P floating-point operations per token but "
        "must read all P weights from memory for every forward pass. With one sequence and one new token per "
        "pass, the ratio of arithmetic to memory traffic (arithmetic intensity) is about one operation per byte "
        f"in 16-bit precision, far below what GPUs need to be compute-bound {cite('roofline')}. Processing k tokens "
        "in one pass multiplies the arithmetic by k but reads the weights once, so until k is large enough to make "
        "the pass compute-bound its time barely grows. Processors with less parallel arithmetic per unit of memory "
        "bandwidth, such as a laptop CPU, reach that point much earlier.")
    R.h2("Speculative decoding")
    R.p("Let x be the text so far and let a drafter propose d = (d<sub>1</sub>, …, d<sub>g</sub>). The model runs "
        "once on x followed by d and returns its own greedy choice y<sub>i</sub> after every prefix. The longest "
        "prefix with d<sub>i</sub> = y<sub>i</sub> is accepted and the model's token at the first mismatch is "
        "appended, so every pass yields between 1 and g + 1 tokens. Writing τ(g) for the mean number of tokens "
        "gained per pass and s(g) for the time of a pass that verifies g drafted tokens, throughput and speedup "
        "over normal decoding are")
    R.equation(r"\mathrm{throughput}(g)=\frac{\tau(g)}{s(g)},\qquad \mathrm{speedup}(g)=\frac{\tau(g)\,s(0)}{s(g)},")
    R.p("where s(0) is the time of a normal one-token pass. τ(g) grows with g but saturates, because a single "
        "rejection discards everything after it; s(g) grows with g at a rate set by the hardware. The best draft "
        "size g* maximises Equation 1. τ is a property of the model and the text, s of the device.")
    R.h2("Drafters")
    R.p(f"The original methods {cite('leviathan', 'chen')} draft with a smaller model of the same family, which "
        f"costs memory for a second model and its cache. Medusa {cite('medusa')} and EAGLE {cite('eagle')} train "
        f"extra heads or a light draft network on the target model's features. Lookahead decoding {cite('lookahead')} "
        "generates n-grams by Jacobi iteration inside the model. Retrieval drafters copy continuations from text: "
        f"prompt lookup decoding {cite('pld')} searches the prompt and the output so far, and REST {cite('rest')} "
        f"searches an external datastore. SpecInfer {cite('specinfer')} verifies a tree of alternative drafts in one "
        f"pass, and Sequoia {cite('sequoia')} sizes such trees for a given device. Xia et al. {cite('specbench')} "
        "survey the field and provide the Spec-Bench evaluation suite. Training-free retrieval drafting needs no "
        "extra model or memory, which makes it the natural choice for small devices; its limit is that it can only "
        "propose text that already exists in the context.")
    R.h2("UniSpec")
    R.p(f"UniSpec {cite('unispec')} is a training-free n-gram method. Its device-aware calibration runs the model on "
        "the target device for m draft sizes in a range, records the pass time s<sub>i</sub> and the acceptance "
        "τ<sub>i</sub> for each, fits regression models f<sub>time</sub>(g) ≈ s(g) and f<sub>accept</sub>(g) ≈ τ(g), "
        f"and chooses g* = argmax f<sub>accept</sub>(g)/f<sub>time</sub>(g) with Differential Evolution {cite('de')}. "
        "Two further components assign confidence scores to n-grams from the verifier's probabilities and expand a "
        "draft tree with a broader first level and threshold pruning. UniSpec reports speedups of up to 2.6× on "
        "four GPUs (A100, A40, RTX A6000, RTX 3090) with 8B–14B models in seven languages, and notes that consumer "
        "GPUs prefer smaller draft sizes than data-centre GPUs. Its stated limitations are that calibration assumes "
        "stable hardware performance and that languages with richer morphology were not evaluated and could yield "
        "lower acceptance.")
    R.h2("Hardware-Aware Parallel Prompt Decoding")
    R.p(f"PPD {cite('ppd')} drafts with trained prompt-token embeddings appended to the input, at about 0.0004% "
        "runtime memory overhead. Its hardware-aware tree pruning chooses the tree size n that maximises "
        "Speedup(n) = τ(n)/L<sub>fp</sub>(n), where the acceptance length τ(n) is treated as hardware-independent "
        "and evaluated once on a validation set, and the forward latency L<sub>fp</sub>(n) is profiled on each "
        "device. Its limitations note that speedups should fall on GPUs with limited compute. Training the "
        "embeddings takes about 16 A100 GPU-hours, beyond this project's budget, so we use PPD's "
        "measure-once/profile-per-device split rather than its drafter.")
    R.h2("The rule both papers share")
    R.p("Both papers choose the draft size from acceptance divided by device latency, and both assume that "
        "acceptance does not depend on the device. Neither tests the rule on CPUs, on free or shared GPUs, under "
        "thermal or power-state changes, or in Indian languages. These are the gaps this project addresses.")

    # ---- 3 Research questions -------------------------------------------------
    R.h1("Research questions")
    R.numbered([
        "<b>Hardware fingerprint.</b> How does the verification cost s(g) differ between a CPU and a GPU?",
        "<b>Device-independence.</b> Is τ(g) the same on every device, as both papers assume?",
        "<b>Optimal draft size.</b> How far apart are the devices' g*, and what does a device lose by using another "
        "device's g*? Can it fall below 1.0×?",
        "<b>Calibration.</b> Do grid search, UniSpec's regression fit and PPD's split choose good draft sizes on "
        "inputs not used for calibration?",
        "<b>Languages.</b> For Hindi, Marathi, Bengali and Tamil, do tokens per word, acceptance and speedup move "
        "together, as our proposal predicted?",
        "<b>Stability.</b> Does a calibrated draft size keep its speedup on a laptop under sustained load?",
    ])

    # ---- 4 Method ------------------------------------------------------------
    R.h1("Methodology")
    R.h2("Decoding engine")
    R.p(f"The engine, written in PyTorch {cite('pytorch')} on Hugging Face Transformers {cite('transformers')}, is "
        "adapted from the one used at the mid-project stage. It maintains one invariant: the key–value cache holds "
        "exactly the tokens that have been verified. After every pass the cache is cropped to the accepted prefix, "
        "so a rejected draft token is never attended to; violating this invariant was the cause of an early bug "
        "in which speculative output drifted from greedy output. Normal greedy decoding runs through the same loop "
        "with an empty drafter, so baseline and speculative timings share one code path. For this study we added "
        "CPU support, automatic precision selection, timing of every forward pass, separate timing of "
        "the prompt prefill, and output logits restricted to the positions that are verified. Each step is:")
    R.code("seq   = prompt;  pending = prompt                  # tokens not yet in the KV cache\n"
           "while generated < max_new:\n"
           "    draft = drafter(seq)[:g]                       # nothing during prefill\n"
           "    preds = model(pending + draft)                 # one pass, greedy choice per position\n"
           "    n     = longest prefix with draft[i] == preds[i]\n"
           "    keep  = draft[:n] + [preds[n]]                 # accepted guesses + model's own token\n"
           "    crop KV cache to the accepted prefix           # rejected guesses leave the cache\n"
           "    seq  += keep;  pending = [seq[-1]]")
    R.p("Before any timing is used, unit tests with a deterministic mock model check that the output equals greedy "
        "decoding token for token for random drafters, hit rates and draft sizes, and that the token budget is "
        "never exceeded.")
    R.h2("Drafter")
    R.p(f"The drafter is a single-chain prompt-lookup n-gram drafter {cite('pld')}. It keeps an incrementally "
        "updated index of every 2-, 3- and 4-gram in the prompt and output, looks up the longest suffix of the "
        "current text that occurred earlier (most recent occurrence first) and proposes up to g tokens that "
        "followed it. UniSpec's confidence scores and draft trees are not implemented; we evaluate its calibration "
        "rule, not its full drafter.")
    R.h2("Choosing the draft size")
    R.p("Three methods are compared. <b>Grid search</b> takes the g with the highest measured speedup on the "
        "calibration articles. The <b>UniSpec fit</b> fits")
    R.equation(r"f_{\mathrm{accept}}(g)=1+a\,(1-e^{-g/b}),\qquad f_{\mathrm{time}}(g)=c_0+c_1g+c_2g^2")
    R.p("to τ and to the mean pass time measured in real decoding on the device, and maximises "
        "f<sub>accept</sub>(g)·s(0)/f<sub>time</sub>(g) over continuous g in the tested range with Differential "
        "Evolution, rounding to the better neighbouring integer. The <b>PPD split</b> takes τ(g) and the mean number "
        "of tokens per pass, <i>k</i><sub>avg</sub>(g), from decoding runs, which may come from a different device, and reads the pass "
        "time from the device's synthetic latency profile s(n), a pass over n new tokens, by interpolation:")
    R.equation(r"\mathrm{speedup}_{\mathrm{PPD}}(g)=\frac{\tau(g)\;s(1)}{s\left(k_{\mathrm{avg}}(g)\right)}")
    R.p("Each chosen g is then measured on held-out articles. When any chosen g lies between the sweep's grid "
        "points, all chosen values are re-measured together in one session, so that no method is compared against "
        "numbers taken under different machine conditions.")
    R.h2("Model, data and protocol")
    R.bullets([
        f"<b>Model.</b> Qwen2.5-0.5B-Instruct {cite('qwen')}, scaled-dot-product attention, batch size 1. It is "
        "small enough to run on any laptop processor and is openly downloadable.",
        f"<b>Data.</b> The XL-Sum {cite('xlsum')} test sets for English, Hindi, Marathi, Bengali and Tamil (BBC "
        "news). From each, 40 articles of at least 300 words are drawn with a fixed seed and truncated to their "
        "first 250 words; alternate articles form a calibration split and a held-out split.",
        "<b>Task.</b> Summarise the article in 4–5 sentences in its own language. A system message (\"Always answer "
        "in {language} only\") keeps the small model from answering Indian-language articles in English. "
        "Summarisation is grounded: the output can reuse input phrases, which an n-gram drafter needs.",
        "<b>Decoding.</b> Greedy, 128 new tokens. After warm-up, every article is decoded with the baseline and "
        "with every draft size back to back, in a shuffled order, and speedups are computed per article and then "
        "averaged. Pairing makes slow drifts in machine state affect all methods equally.",
    ])
    R.p("The prompt, identical for every language apart from the language name, is:")
    R.code("system: You are a helpful assistant. Always answer in {Language} only.\n"
           "user:   Summarize this {Language} news article in {Language} in 4-5 sentences.\n\n"
           "        {first 250 words of the article}")
    dev_rows = []
    for r in runs:
        i = r.info
        mem = f"{i.get('gpu_vram_gb', '')} GB VRAM" if r.kind == "cuda" else f"{i.get('ram_gb', '')} GB RAM"
        cfg = r.exp.get("sweep", r.exp.get("profile"))["config"]
        gmax = max(cfg["grid"])
        cores = f", {i.get('physical_cores')} cores" if r.kind == "cpu" and i.get("physical_cores") else ""
        dev_rows.append([r.label, f"{r.short}{cores}; {mem}", i["dtype"].replace("float", "fp"),
                         f"{cfg['n_calib']} + {cfg['n_eval']}", cfg["repeats"], f"1–{gmax}",
                         f"PyTorch {i['torch'].split('+')[0]}"])
    R.table(["Run", "Hardware", "Precision", "Articles (calib + held-out)", "Repeats", "Draft sizes", "Software"],
            dev_rows,
            "Devices and runs. Each run is one results folder; a second run on the same device under a different "
            "setting is listed separately. CPUs use 32-bit floating point; Turing GPUs such as the T4 use 16-bit "
            "(they lack fast bfloat16).",
            widths=[3.2 * cm, 4.0 * cm, 1.7 * cm, 2.1 * cm, 1.5 * cm, 1.6 * cm, 2.3 * cm])
    R.h2("Metrics and experiments")
    R.p("<b>Correctness</b> is the share of runs whose token ids equal greedy decoding exactly. <b>τ</b> is generated "
        "tokens after the first divided by decode passes. <b>Speedup</b> is baseline latency over speculative "
        "latency for the same article; <i>end to end</i> includes reading the prompt (prefill), <i>generation-only</i> "
        "excludes it, since speculation cannot accelerate prefill. <b>Copy share</b> is the fraction of the summary's "
        "4-token sequences that also occur in the article; <b>self-repeat</b> is the fraction that occurred earlier in "
        "the summary itself. <b>Tokens per word</b> is tokenizer pieces per whitespace-separated word. The "
        "experiments are:")
    R.numbered([
        "correctness at g = 2, 8 and 32;",
        "a latency profile: one forward pass of n = 1 to 33 new tokens (to 129 in the extended GPU run) over a cache "
        "of 512, 1,024 and 2,048 tokens, median of 7–15 timings each;",
        "a draft-size sweep over the grid in Table 1;",
        "calibration with the three methods and a held-out check;",
        "a language study: tokens per word for the Qwen2.5 and Sarvam-1 tokenizers, then τ and speedup at several "
        "draft sizes on held-out articles in each language;",
        "a sustained-load test on the laptop: 20 minutes of continuous rounds, each timing a one-token and a "
        "(1 + g*)-token pass and summarising one held-out article with and without speculation.",
    ])

    # ---- 5 Results -----------------------------------------------------------
    R.h1("Results")
    corr_txt = []
    for r in runs:
        if r.has("correctness"):
            c = r.exp["correctness"]["exact_match_rate"]
            if r.sweep:
                lo, hi = rd.sweep_exact_range(r)
                corr_txt.append(f"{r.label}: {pct(c)} in the correctness check and {pct(lo)}–{pct(hi)} across the "
                                f"sweep" if lo < 1 else f"{r.label}: {pct(c)} in the correctness check and in every "
                                f"sweep run")
    R.p("Exact agreement with greedy decoding was: " + "; ".join(corr_txt) + ". On 16-bit devices the few "
        "divergent runs split from greedy output at a near-tie between two tokens, where batched and sequential "
        "evaluation round differently; the same effect was traced at the mid-project stage. It is a property of "
        "half-precision arithmetic, not of the method, and does not occur in 32-bit arithmetic.")

    R.h2("Hardware fingerprint")
    prof_rows = []
    for r in runs:
        if not r.has("profile"):
            continue
        p = rd.profile_at(r)
        cell = lambda n: f"{p[n]:.1f}" if n in p else "–"  # noqa: E731
        prof_rows.append([r.label, cell(1), cell(2), cell(9), cell(33), cell(129),
                          x(p[33] / p[1]) if 33 in p else "–"])
    R.table(["Run", "1 token (ms)", "2 tokens", "9 tokens", "33 tokens", "129 tokens", "33 vs 1 token"], prof_rows,
            "Time of one forward pass by the number of new tokens it processes, at a 1,024-token context "
            "(median of repeated timings).",
            widths=[4.2 * cm, 2.0 * cm, 1.8 * cm, 1.8 * cm, 1.9 * cm, 2.0 * cm, 2.6 * cm])
    fp_sent = []
    for r in mains:
        p = rd.profile_at(r)
        fp_sent.append(f"on the {r.label} a one-token pass takes {p[1]:.0f} ms, two tokens {x(p[2] / p[1])} and "
                       f"33 tokens {x(p[33] / p[1])} as long")
    R.p("The devices sit at opposite ends of the cost curve (Figure 1, Table 2): " + "; ".join(fp_sent) + ". "
        f"The {gpu.short} remains almost flat up to {gpu_maxn} tokens ({p_gpu[gpu_maxn]:.1f} ms), so verifying "
        "even a hundred extra tokens is nearly free for a model of this size. On the CPU the first extra token is "
        f"the most expensive: two tokens cost {x(p_cpu[2] / p_cpu[1])} one token, after which each further token adds "
        f"about {(p_cpu[33] - p_cpu[2]) / 31 / p_cpu[1] * 100:.0f}% of a one-token pass. The likely reason is a change "
        "of kernel: a single token is a matrix–vector product, while two or more tokens use general matrix–matrix "
        "routines with a higher fixed cost.")
    R.figure(fig("fig1_hardware_fingerprint.png"),
             "Hardware fingerprint. Left: time of one forward pass against the number of tokens processed. Right: the "
             "same relative to a one-token pass. A flat line means verifying drafted tokens is nearly free.")

    R.h2("Acceptance is device-independent")
    tau_rows = []
    for g in tau_gs:
        tau_rows.append([g] + [f"{tau_aggs[r.folder][g]['tau']:.2f}" for r in tau_runs])
    R.table(["g"] + [f"τ, {r.label}" for r in tau_runs], tau_rows,
            f"Tokens gained per pass, τ(g), on the {len(common)} articles summarised in every run.",
            widths=[1.2 * cm] + [(W - 1.2 * cm) / len(tau_runs)] * len(tau_runs))
    R.p(f"On the {len(common)} articles summarised in every run, τ agrees to within {tau_gap:.2f} at every draft "
        "size (Table 3, Figure 2 left), although the devices differ in architecture, precision and speed by more "
        "than a factor of four. The remaining differences come from the half-precision near-ties above, which "
        "occasionally change the text and with it the acceptance. This confirms the assumption behind both "
        "UniSpec's calibration and PPD's split: acceptance can be measured once, and only latency needs to be "
        "measured per device.")

    R.h2("Optimal draft size and the cost of a wrong one")
    show_g = sorted({g for r in swept for g in r.sweep if g > 0} & {1, 2, 4, 8, 16, 32, 64, 128})
    sp_rows = []
    for r in swept:
        best = rd.best_g(r)[0]
        sp_rows.append([r.label] + [(f"<b>{x(r.sweep[g]['speedup'])}</b>" if g == best else x(r.sweep[g]["speedup"]))
                                    if g in r.sweep else "–" for g in show_g])
    R.table(["Run"] + [f"g = {g}" for g in show_g], sp_rows,
            "Mean end-to-end speedup over normal decoding by draft size, all articles. Bold: the best draft size of "
            "each run.", widths=[3.7 * cm] + [(W - 3.7 * cm) / len(show_g)] * len(show_g))
    best_sent = []
    for r in swept:
        g, v = rd.best_g(r)
        top = max(g2 for g2 in r.sweep)
        best_sent.append(f"the {r.label} peaks at g = {g} ({x(v)})" + (", the largest size tried" if g == top else ""))
    R.p("Because τ is shared and s(g) is not, the devices want very different draft sizes (Table 4, Figure 2): "
        + "; ".join(best_sent) + ". "
        f"On the CPU the speedup falls to {x(cpu_g32)} at g = 32, and for the "
        f"{len(slow32)} articles that copied least from their source g = 32 was slower than normal decoding "
        f"({', '.join(x(v) for v in slow32)}). On the {gpu.short} the curve is still rising at the largest draft "
        "size tested: with a 0.5B model the GPU never runs out of spare capacity in this range, so its true optimum "
        "may be higher still.")
    R.figure(fig("fig2_draft_size_sweep.png"),
             "Left: τ on the articles run on every device; the curves coincide. Right: measured end-to-end speedup "
             "by draft size; stars mark each run's calibrated g*. The dashed line is normal decoding.")
    wr_rows = []
    for t in wrong:
        for o, v in t["others"].items():
            wr_rows.append([t["device"], t["own_g"], x(t["own"]), f"{v['g']} ({o})", x(v["speedup"]),
                            pct(1 - v["speedup"] / t["own"])])
    R.table(["Run", "Own g*", "Speedup", "g* calibrated elsewhere", "Speedup", "Loss"], wr_rows,
            "Speedup on held-out articles with each run's own calibrated g* and with a g* calibrated in another run, "
            "read from the run's own sweep (interpolated between tested draft sizes; draft sizes beyond a run's "
            "tested range are omitted).",
            widths=[3.7 * cm, 1.4 * cm, 1.6 * cm, 5.4 * cm, 1.6 * cm, 1.4 * cm])
    R.p("Using a draft size calibrated elsewhere costs speed on every run (Table 5, Figure 3). The CPU loses most, "
        "because the GPU's large draft size sits on the steep part of the CPU's cost curve; the GPU loses less "
        "because a small draft size only forgoes acceptance it could have had for free. Averaged over articles no "
        "run fell below 1.0× with a foreign draft size, but individual articles did on the CPU, as noted above.")
    R.figure(fig("fig3_wrong_g_cost.png"),
             "Cost of using the wrong draft size: held-out speedup with each run's own g* (dark) and with g* "
             "calibrated in another run (light).", width=W * 0.92)

    R.h2("Choosing the draft size automatically")
    cal_rows = []
    for r in runs:
        if not r.has("calibration"):
            continue
        for k, ch in r.exp["calibration"]["choices"].items():
            pred = ch.get("pred_speedup")
            cal_rows.append([r.label, rd.method_name(k, runs), ch["g_star"],
                             x(pred) if pred == pred and pred is not None else "–", x(ch["heldout_speedup"])])
    R.table(["Run", "Method", "g*", "Predicted", "Measured (held-out)"], cal_rows,
            "Draft size chosen by each method, the speedup it predicted from the calibration articles, and the "
            "speedup measured on held-out articles.",
            widths=[3.9 * cm, 6.0 * cm, 1.2 * cm, 2.0 * cm, 3.3 * cm])
    cal_cpu = cpu.exp["calibration"]["choices"]
    R.p(f"On the {cpu.label}, measured together in one session on held-out articles, the methods chose g between "
        f"{min(c['g_star'] for c in cal_cpu.values())} and {max(c['g_star'] for c in cal_cpu.values())} and delivered "
        f"{x(min(c['heldout_speedup'] for c in cal_cpu.values()))}–{x(max(c['heldout_speedup'] for c in cal_cpu.values()))}, "
        "within about 10% of each other, because the CPU's speedup curve is nearly flat around its optimum (Table 6, "
        f"Figure 4). On the {gpu.short}, every method in the first sweep, including the PPD split using τ measured "
        "on the CPU, chose the largest draft size tested, and in the extended sweep UniSpec's fit and grid search "
        "again chose the largest size.")
    R.p("Two effects stand out. First, every method over-predicts the speedup, by 14% for grid search on the CPU and "
        "by up to 2.7 times for the UniSpec fit on the extended GPU sweep. Section 5.5 shows why: the speedup depends "
        "strongly on how much each article is copied, and the calibration articles happened to copy more than the "
        "held-out ones. Second, the PPD split relies on one short synthetic latency profile. On the shared cloud GPU "
        "a few profile readings jumped from about 28 to 40 ms; combined with τ from the CPU, whose sweep stopped at "
        "g = 32, the split then chose g = 7 and lost about 18% against the fit. Methods that time real decoding were "
        "not affected.")
    R.figure(fig("fig4_calibration_methods.png"),
             "Calibration methods per run: speedup predicted from the calibration articles (light) and measured on "
             "held-out articles (dark), with the chosen g*.")

    R.h2("Copying drives the speedup")
    corr_sent = []
    for r in swept:
        c, n = rd.copy_correlation(r)
        if c is not None:
            corr_sent.append(f"r = {c:.2f} over {n} articles on the {r.label}")
    sc = rd.split_copy(cpu)
    R.p("Across articles the speedup varies widely, and the share of the summary copied from the article explains "
        "most of that variation (Figure 5). At g = 8 the Pearson correlation between copy share and "
        "generation-only speedup is " + "; ".join(corr_sent) + ". On the CPU, English summaries that copy about 20% "
        "of their 4-token sequences run about 1.1× faster, while those that copy 90% run about 3.4× faster. The "
        f"calibration articles copied {pct(sc['calib'])} on average and the held-out articles {pct(sc['eval'])}, "
        "consistent with the over-prediction in Section 5.4. For a retrieval drafter, the task decides how much "
        "can be gained and the hardware decides how large g can usefully be.")
    R.figure(fig("fig7_copying_vs_speedup.png"),
             "Generation-only speedup at g = 8 against the share of the summary's 4-token sequences that also "
             "occur in the article; one point per article, marker shape by language.")

    R.h2("Indian languages")
    fert_rows = []
    any_lang = next(iter(lang_by_run.values()))
    ptoks = {x_["lang"]: x_["prompt_tokens"] for x_ in any_lang}
    tok_names = list(fert)
    for l in rd.LANG_ORDER:
        fert_rows.append([l.capitalize()] + [f"{fert[t][l]['tokens_per_word']:.2f}" for t in tok_names]
                         + [f"{ptoks[l]:,.0f}"])
    R.table(["Language"] + [f"Tokens per word, {t.split('/')[-1]}" for t in tok_names] + ["Prompt length (tokens)"],
            fert_rows, "Tokenizer granularity on the 250-word article excerpts, and the resulting mean prompt length "
            "for Qwen2.5 (including instructions).",
            widths=[2.6 * cm] + [4.2 * cm] * len(tok_names) + [3.6 * cm])
    R.p(f"Qwen2.5's tokenizer splits the Indian languages into {min(ind_fert):.1f}–{max(ind_fert):.1f} tokens per "
        f"word against {en_fert:.2f} for English (Table 7). The tokenizer of Sarvam-1, a model built for Indian "
        "languages, needs only 1.4–2.3 tokens per word for the same text, so the difference lies in the tokenizer "
        "rather than in the languages. As a result a 250-word Tamil excerpt becomes a prompt about seven times longer "
        "than its English counterpart.")
    lang_rows = []
    for r in mains:
        for row in lang_by_run.get(r.folder, []):
            lang_rows.append([f"{r.short}, g = {row['g']}", row["lang"].capitalize(), pct(row["copy"]), pct(row["rep"]),
                              f"{row['tau']:.2f}", x(row["gen"]), x(row["e2e"]), f"{row['prefill']:.2f}"])
    R.table(["Run", "Language", "Copied", "Repeated", "τ", "Generation only", "End to end", "Prefill (s)"],
            lang_rows, "Language study on held-out articles at each device's calibrated draft size. Copied and repeated: share "
            "of the summary's 4-token sequences found in the article and earlier in the summary. Prefill is the "
            "time normal decoding spends reading the prompt.",
            widths=[3.4 * cm, 1.7 * cm, 1.4 * cm, 1.8 * cm, 1.2 * cm, 2.6 * cm, 2.1 * cm, 1.9 * cm])
    R.p("Our proposal predicted that more pieces per word would make tokens harder to guess and lower acceptance. "
        "The opposite happened (Table 8, Figure 6). Two effects combine. The 0.5B model largely copies the article "
        f"when summarising in these languages: {pct(min(min(x_['copy'] for x_ in v if x_['lang'] in ind) for v in lang_by_run.values()))}–"
        f"{pct(max(max(x_['copy'] for x_ in v if x_['lang'] in ind) for v in lang_by_run.values()))} of its 4-token "
        "sequences come from the article, against about half in English, while self-repetition stays low. And each "
        "copied word is many tokens long, so a single correct guess of a word yields many tokens at once.")
    tam_c, tam_g = cpu_lang["tamil"], gpu_lang["tamil"]
    R.p(f"How much of that gain survives end to end depends on how fast the device reads the prompt. On the "
        f"{gpu.short}, generation alone became {x(min(gpu_lang[l]['gen'] for l in ind))}–"
        f"{x(max(gpu_lang[l]['gen'] for l in ind))} faster for the Indian languages and whole requests "
        f"{x(min(gpu_lang[l]['e2e'] for l in ind))}–{x(max(gpu_lang[l]['e2e'] for l in ind))} faster. On the "
        f"{cpu.short}, generation alone was {x(min(cpu_lang[l]['gen'] for l in ind))}–"
        f"{x(max(cpu_lang[l]['gen'] for l in ind))} faster, but whole requests only "
        f"{x(min(cpu_lang[l]['e2e'] for l in ind))}–{x(max(cpu_lang[l]['e2e'] for l in ind))}. Reading a Tamil prompt "
        f"takes {tam_c['prefill']:.1f} s of a {tam_c['latency']:.1f} s request on the CPU but {tam_g['prefill']:.2f} s "
        "on the GPU. Fine-grained tokenization thus helps acceptance and hurts prefill, and the hardware decides which "
        "effect dominates.")
    R.p("The model's summaries in these languages are of low quality and heavily extractive. Decoding is lossless, "
        "so this does not affect the speed comparison, but a larger model that paraphrases more would likely accept "
        "fewer drafted tokens.")
    R.figure(fig("fig5_language.png"),
             "Languages. Left: tokens per word for two tokenizers. Middle: τ at each device's calibrated draft size. "
             "Right: speedup at that draft size, end to end (solid) and generation only (dashed).")

    R.h2("Sustained load")
    for r in mains:
        if not r.has("thermal"):
            continue
        g, cyc = rd.thermal_cycles(r)
        R.table(["Cycle", "Minutes", "1-token pass (ms)", f"{1 + g}-token pass (ms)", "Normal decoding (tokens/s)",
                 f"Speedup at g = {g}"],
                [[i + 1, f"{c['t0']:.1f}–{c['t1']:.1f}", f"{c['p1']:.0f}", f"{c['pg']:.0f}", f"{c['tps']:.1f}",
                  x(c["spd"])] for i, c in enumerate(cyc)],
                f"Sustained load on the {r.label}: each cycle covers every held-out article once, so article content "
                "averages out.", widths=[1.3 * cm, 2.1 * cm, 2.8 * cm, 2.9 * cm, 3.6 * cm, 3.4 * cm])
        R.p(f"Over {cyc[-1]['t1']:.0f} minutes of continuous work on the plugged-in {r.label}, pass times and the "
            f"speedup at the calibrated g = {g} showed no downward trend: the per-cycle speedup stayed between "
            f"{x(min(c['spd'] for c in cyc))} and {x(max(c['spd'] for c in cyc))} (Table 9, Figure 7). The raw "
            "per-round speedup varies between about 1.0× and 2.1×, but that is the articles taking turns, which is "
            "why the table averages over full cycles.")
    R.p("Power state mattered more than heat. In a separate short run on battery power the same laptop needed about "
        "197 ms per normal pass instead of 124 ms. A draft size calibrated on mains power is then applied to a "
        "different cost curve, which is the kind of instability UniSpec's limitations anticipate. Our software "
        "therefore detects battery operation and stores such runs separately.")
    R.figure(fig("fig6_thermal.png"),
             "Sustained load. Left: one-token and (1 + g*)-token pass times over 20 minutes. Right: speedup at the "
             "calibrated g*, averaged per cycle of held-out articles.")

    # ---- 6 Discussion --------------------------------------------------------
    R.h1("Discussion")
    R.p("<b>The shared rule holds and its split is practical.</b> Acceptance matched across a 32-bit CPU and a 16-bit "
        "GPU, so τ can be measured once on any machine and only the latency curve, a two-minute measurement, has to "
        "be repeated per device. This is the practical value of PPD's decomposition for anyone without long access to "
        "the target hardware.")
    R.p("<b>Everyday hardware sits at the far end of the cost curve.</b> UniSpec observes smaller optimal draft sizes "
        "on consumer GPUs than on data-centre GPUs. The laptop CPU extends that trend: beyond the first extra token, "
        "verification cost grows almost linearly with the draft size, the optimum falls to around 8–16, and a draft "
        "size suited to a GPU loses 14% and can make individual requests slower than no speculation at all. The free "
        "T4, in contrast, still had idle capacity at g = 128, because a 0.5B model leaves it memory-bound; with "
        "UniSpec's 8B–14B models its optimum would be lower.")
    R.p("<b>Calibration selects well but predicts poorly.</b> Because the speedup curves are flat near their optimum, "
        "every reasonable method chose a draft size close to the best. The predicted speedups, however, ranged from "
        "14% too high to 2.7 times the measured value, because acceptance depends on how much each input is copied "
        "and a handful of calibration articles does not represent that well. A deployment should calibrate on inputs "
        "typical of its own workload and treat the predicted speedup as an upper estimate.")
    R.p("<b>Profile-only calibration needs repeated measurements on shared hardware.</b> The PPD split rests on a "
        "single synthetic latency profile. On a shared cloud GPU, transient slowdowns in a few readings moved its "
        "choice from the largest draft size to g = 7. Repeating the profile and taking the median per size, or timing "
        "real decoding as UniSpec does, avoids this.")
    R.p("<b>Tokenization decides the language result.</b> Fine-grained tokenization of Indian scripts is not a barrier "
        "to n-gram speculation: copied words become long, easily guessed runs of tokens. It is a barrier to latency on "
        "devices with slow prefill, because the same text costs several times more tokens to read. A tokenizer built "
        "for these languages, such as Sarvam-1's, would shorten prompts by a similar factor and remove most of that "
        "cost.")

    # ---- 7 Limitations -------------------------------------------------------
    R.h1("Limitations")
    R.bullets([
        "<b>Simplified UniSpec.</b> A single chain of drafted tokens replaces confidence-scored draft trees; we "
        "evaluate UniSpec's calibration, not its full drafter, so absolute speedups are not comparable with the paper.",
        "<b>No PPD drafter.</b> Its prompt-token embeddings require GPU training beyond our budget.",
        "<b>One small model.</b> Qwen2.5-0.5B is far smaller than the papers' 7B–14B models. Larger models load the "
        "GPU more and would lower its optimal draft size; on a CPU they would make every pass slower.",
        f"<b>Few devices.</b> {n_dev} device{'s' if n_dev != 1 else ''} so far. The benchmark runs unchanged on further "
        "CPUs and GPUs, and their results join every table and figure of this report on rebuilding.",
        "<b>Timing noise.</b> Laptop and shared-GPU timings are noisier than dedicated servers. Paired back-to-back runs, "
        "held-out articles and single-session comparisons reduce this, but differences below about 5% should not be "
        "over-interpreted.",
        "<b>Indian-language quality.</b> The small model copies heavily in these languages; acceptance is likely lower "
        "with stronger models.",
    ])

    # ---- 8 Conclusion ---------------------------------------------------------
    R.h1("Conclusion")
    R.p("The hardware-aware rule shared by UniSpec and PPD transfers to everyday hardware: acceptance is "
        "device-independent, latency is not, and measuring latency per device selects a good draft size. The optimal "
        f"draft size differs by an order of magnitude between a laptop CPU ({cpu_bg}) and a free cloud GPU "
        f"({gpu_bg} or more), and using the wrong one costs 6–14% on average and can make individual requests slower "
        "than normal decoding. Calibration chooses well but over-promises, because the gain depends as much on how much "
        "the output copies its input as on the hardware. In Indian languages, fine-grained tokenization raises "
        f"acceptance, up to {x(max(gpu_lang[l]['e2e'] for l in ind))} end to end on the GPU, but lengthens prompts so "
        f"much that on a CPU reading them limits the gain to {x(min(cpu_lang[l]['e2e'] for l in ind))}–"
        f"{x(max(cpu_lang[l]['e2e'] for l in ind))}.")

    # ---- References ------------------------------------------------------------
    R.h1("References", numbered=False)
    for i, (_, text) in enumerate(REFS, 1):
        R.p(f"[{i}]&nbsp;&nbsp;{text}", "ref")

    # ---- Appendices ------------------------------------------------------------
    R.story.append(PageBreak())
    R.appendix("A", "Draft-size sweeps in full")
    for r in swept:
        rows = [[g, f"{v['tau']:.2f}", pct(v["acceptance_rate"]), f"{v['mean_pass_len']:.1f}",
                 f"{v['mean_pass_sec'] * 1000:.1f}", x(v["speedup"]), x(v["decode_speedup"]), pct(v["exact_match"])]
                for g, v in r.sweep.items()]
        R.table(["g", "τ", "Accepted", "Tokens per pass", "Pass time (ms)", "Speedup", "Generation-only",
                 "Exact match"], rows, f"Draft-size sweep on the {r.label}, all articles (g = 0 is normal decoding).",
                widths=[1.2 * cm, 1.5 * cm, 1.9 * cm, 2.4 * cm, 2.4 * cm, 2.0 * cm, 2.6 * cm, 2.4 * cm])
    R.appendix("B", "Fitted calibration models")
    fit_rows = []
    for r in runs:
        u = r.exp.get("calibration", {}).get("choices", {}).get("unispec")
        if not u:
            continue
        tf, mf = u["tau_fit"], u["time_fit"]["coef"]
        fit_rows.append([r.label, f"{tf['a']:.2f}", f"{tf['b']:.2f}", f"{mf[2] * 1000:.2f}", f"{mf[1] * 1000:.3f}",
                         f"{mf[0] * 1000:.5f}", f"{u['g_continuous']:.1f}", u["g_star"]])
    R.table(["Run", "a", "b", "c<sub>0</sub> (ms)", "c<sub>1</sub> (ms)", "c<sub>2</sub> (ms)", "Continuous g*",
             "Integer g*"], fit_rows,
            "Parameters of the UniSpec-style fits of Equation 2 on the calibration articles. The continuous optimum "
            "is found by Differential Evolution and rounded to the better neighbouring integer.",
            widths=[3.8 * cm, 1.3 * cm, 1.3 * cm, 1.7 * cm, 1.7 * cm, 2.0 * cm, 2.2 * cm, 2.0 * cm])
    R.appendix("C", "Reproducing and extending the results")
    R.p(f"Code, the fixed article subset, raw results, figures and this report's generator are in the project "
        f"repository: {REPO}. On any machine with Python 3.9 or newer:")
    R.code("pip install -r requirements.txt\n"
           "python tests/test_engine.py            # engine correctness with a mock model\n"
           "python run.py all                      # every benchmark on this machine\n"
           "python run.py thermal --minutes 20     # sustained load (laptops)\n"
           "python run.py plots                    # figures and results/SUMMARY.md\n"
           "python report/build_report.py          # this report, rebuilt from results/")
    R.p("The device, number precision and workload are detected automatically, and a laptop on battery is recorded "
        "as a separate run. To add a device, run the benchmark on it, copy its results folder into results/, and "
        "rebuild the figures and this report: every table and figure includes the new run. GPU results here were "
        "produced with notebooks/colab_t4.ipynb on Google Colab's free T4.")
    doc = Doc(out_path)
    doc.multiBuild(R.story)
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--figures", default=os.path.join(ROOT, "figures"))
    ap.add_argument("--out", default=os.path.join(HERE, "report.pdf"))
    ap.add_argument("--plots", action="store_true", help="regenerate figures from results/ first")
    a = ap.parse_args()
    if a.plots:
        from specdec import plots
        plots.make_all(a.results, a.figures)
    build(a.results, a.figures, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
