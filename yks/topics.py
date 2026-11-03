"""Shared YKS curriculum taxonomy (subject -> topics), used by data generation, diagnosis and mastery."""

TOPICS = {
    "Mathematics": [
        "temel kavramlar ve sayı basamakları", "bölme-bölünebilme, EBOB-EKOK", "rasyonel sayılar ve ondalık sayılar",
        "üslü ve köklü ifadeler", "çarpanlara ayırma", "oran-orantı", "sayı, kesir, yaş, işçi, hız problemleri",
        "yüzde, kâr-zarar, karışım problemleri", "kümeler ve kartezyen çarpım", "fonksiyonlar ve bileşke",
        "polinomlar", "ikinci dereceden denklemler", "eşitsizlikler ve mutlak değer", "permütasyon ve kombinasyon",
        "binom açılımı", "olasılık", "logaritma", "diziler (aritmetik ve geometrik)", "trigonometri",
        "limit ve süreklilik", "türev ve uygulamaları", "integral", "karmaşık sayılar", "analitik geometri (doğru)",
        "çember ve daire (metinle tanımlı)", "üçgende açı ve uzunluk (metinle tanımlı)",
    ],
    "Physics": [
        "birim ve vektörler", "düzgün doğrusal ve ivmeli hareket", "Newton'un hareket yasaları", "iş, güç ve enerji",
        "itme ve momentum", "basınç ve kaldırma kuvveti", "ısı, sıcaklık ve genleşme", "elektrik akımı, direnç ve devreler",
        "elektrik enerjisi ve güç", "manyetizma ve indüksiyon", "dalgalar", "optik: yansıma ve kırılma",
        "çembersel hareket ve kütle çekimi", "basit harmonik hareket", "modern fizik ve fotoelektrik olay",
    ],
    "Chemistry": [
        "atom modelleri ve periyodik sistem", "kimyasal türler arası etkileşimler", "mol kavramı ve kimyasal hesaplamalar",
        "kimyasal tepkimeler ve denkleştirme", "gazlar", "çözeltiler ve derişim", "kimyasal tepkimelerde enerji",
        "tepkime hızı", "kimyasal denge", "asitler, bazlar ve pH", "çözünürlük dengesi", "elektrokimya",
        "organik kimya: hidrokarbonlar ve fonksiyonel gruplar", "maddenin halleri",
    ],
}


def all_topics() -> list[str]:
    return topics_for("yks")

OLYMPIAD_TOPICS = {
    "Algebra": ["polynomials and Vieta", "inequalities", "functional equations", "sequences and series",
                "complex numbers", "logarithms and exponentials", "systems of equations"],
    "Combinatorics": ["counting and bijections", "probability and expected value", "recursion",
                      "pigeonhole and extremal", "graphs and games", "inclusion-exclusion"],
    "Geometry": ["triangle centers and similarity", "circles and power of a point", "coordinate and vectors",
                 "trigonometry in geometry", "areas and lengths", "3D geometry"],
    "Number Theory": ["divisibility and primes", "modular arithmetic", "digits and bases",
                      "diophantine equations", "gcd, lcm and orders"],
}


def topics_for(dataset: str = "yks") -> list[str]:
    tax = TOPICS if dataset == "yks" else OLYMPIAD_TOPICS
    return [f"{s} / {t}" for s, ts in tax.items() for t in ts]
