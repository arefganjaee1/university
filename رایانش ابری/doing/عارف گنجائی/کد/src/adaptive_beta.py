from dataclasses import dataclass

def clip(x, lo, hi):
    return max(lo, min(hi, x))

@dataclass
class AdaptiveBetaConfig:
    # برای پایداری عددی و جلوگیری از رفتار افراطی
    beta_min: float = 0.05
    beta_max: float = 1.0

    # کنترل drift با KL
    kl_target: float = 0.05
    kl_tolerance: float = 0.02
    alpha_kl: float = 0.15  # شدت واکنش به KL

    # کنترل “اطمینان مدل” با اختلاف logprob
    alpha_conf: float = 0.10

    # روند زمانی (می‌تونه محل نوآوری تجربی باشه)
    warmup_steps: int = 200
    max_steps: int = 2000


def time_schedule(step: int, cfg: AdaptiveBetaConfig) -> float:
    """
    ramp ساده: اول ملایم، بعد قوی‌تر.
    (بعداً می‌تونی cosine/exponential/learned کنی = نوآوری)
    """
    if step <= cfg.warmup_steps:
        return 0.3
    r = (step - cfg.warmup_steps) / max(1, (cfg.max_steps - cfg.warmup_steps))
    r = clip(r, 0.0, 1.0)
    return 0.3 + 0.7 * r  # از 0.3 به 1.0


def update_beta(
    beta_prev: float,
    logp_diff: float,
    kl_value: float,
    step: int,
    cfg: AdaptiveBetaConfig
) -> float:
    """
    ورودی‌ها:
      logp_diff = log p(y_w|x) - log p(y_l|x)  (هرچه بزرگ‌تر → مدل مطمئن‌تر)
      kl_value  = KL(π || π_ref)              (هرچه بزرگ‌تر → drift بیشتر)

    منطق:
      - KL زیاد ⇒ beta کم شود (محافظه‌کارانه‌تر)
      - logp_diff کوچک ⇒ beta کمی زیاد شود (فشار ترجیح بیشتر برای یادگیری)
      - beta در [beta_min, beta_max] clip می‌شود.
    """

    # 1) term مربوط به KL (بازخورد منفی برای drift)
    kl_err = kl_value - cfg.kl_target
    if abs(kl_err) <= cfg.kl_tolerance:
        kl_term = 0.0
    else:
        kl_term = -cfg.alpha_kl * kl_err  # KL بالا → beta پایین

    # 2) term مربوط به اطمینان (بازخورد مثبت وقتی diff کوچک است)
    # اگر logp_diff نزدیک صفر باشد ⇒ 1/(1+0)=1 ⇒ beta بیشتر فشار می‌آورد
    conf_term = cfg.alpha_conf * (1.0 / (1.0 + max(logp_diff, 0.0)))

    # 3) وزن‌دهی زمانی
    w = time_schedule(step, cfg)

    beta_new = beta_prev + w * (kl_term + conf_term)

    # 4) محدودسازی برای پایداری
    beta_new = clip(beta_new, cfg.beta_min, cfg.beta_max)
    return beta_new