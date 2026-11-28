# 1. SETUP & CONFIGURATION
library(readxl)
library(dplyr)
library(tidyr)
library(purrr)
library(ggplot2)
library(ggpubr)
library(cowplot)
library(ggokabeito)
library(ggdist)
library(showtext)
library(sysfonts)
library(scales)

font_add_google("Tinos", "Tinos")
showtext_auto()
showtext_opts(dpi = 150)

theme_set(
  theme_cowplot(font_size = 12, font_family = "Tinos") +
    theme(legend.position = "top")
)

options(
  ggplot2.discrete.colour = palette_okabe_ito(),
  ggplot2.discrete.fill   = palette_okabe_ito()
)

DATA_FILE <- "data/clusters.xlsx"
OUT_DIR   <- "data"

COHORT_LEVELS <- c("none", "WCLS", "WSLC")
FITTED_LEVELS <- c("WCLS", "WSLC")
MOVE_LABELS   <- c(`1` = "Rock", `2` = "Paper", `3` = "Scissors")
SUBMODEL_MAP  <- c(WSLC_I = "WSLC-Imitate", WSLC_C = "WSLC-Counter")

STAT_LABELS <- c(
  win_stay_rate  = "Win-Stay Rate",
  lose_stay_rate = "Lose-Stay Rate",
  counter_rate   = "Counter Rate",
  cramers_v      = "Cramér's V"
)

COHORT_MODEL_MAP <- list(
  none = "Baseline",
  WCLS = "WCLS",
  WSLC = c("WSLC_I", "WSLC_C")
)

MODEL_TO_COHORT <- setNames(
  rep(names(COHORT_MODEL_MAP), lengths(COHORT_MODEL_MAP)),
  unlist(COHORT_MODEL_MAP)
)

# 2. DATA LOADING & PREPARATION
sheets <- excel_sheets(DATA_FILE) %>%
  set_names() %>%
  map(~ read_excel(DATA_FILE, sheet = .x))

sheets[c("comparison", "agreement")] <- map(
  sheets[c("comparison", "agreement")],
  ~ mutate(.x, cohort = factor(cohort, levels = COHORT_LEVELS))
)

sheets[c("fitted_comparison", "fitted_agreement")] <- map(
  sheets[c("fitted_comparison", "fitted_agreement")],
  ~ mutate(.x, cohort = factor(cohort, levels = FITTED_LEVELS))
)

add_stat_label <- function(df) {
  mutate(
    df,
    stat_label = factor(STAT_LABELS[as.character(stat)], levels = STAT_LABELS)
  )
}

to_long <- function(df) {
  stat_pattern <- paste0(
    "^(", paste(names(STAT_LABELS), collapse = "|"), ")_(real|model)$"
  )

  df %>%
    pivot_longer(
      matches(stat_pattern),
      names_to = c("stat", ".value"),
      names_pattern = "(.+)_(real|model)"
    ) %>%
    mutate(stat = factor(stat, levels = names(STAT_LABELS))) %>%
    add_stat_label()
}

matched_long <- sheets$comparison %>%
  rename(strategy = model) %>%
  filter(MODEL_TO_COHORT[strategy] == as.character(cohort)) %>%
  to_long()

cohort_counts <- matched_long %>%
  filter(stat == names(STAT_LABELS)[1]) %>%
  distinct(player, session, cohort) %>%
  count(cohort)

COHORT_LABELS <- setNames(
  paste0(cohort_counts$cohort, " (n=", cohort_counts$n, ")"),
  cohort_counts$cohort
)

add_cohort_label <- function(df) {
  mutate(
    df,
    cohort_label = factor(
      COHORT_LABELS[as.character(cohort)],
      levels = COHORT_LABELS
    )
  )
}

matched_long <- add_cohort_label(matched_long)
fitted_long  <- sheets$fitted_comparison %>%
  to_long() %>%
  add_cohort_label()

wslc_submodel_long <- fitted_long %>%
  filter(cohort == "WSLC") %>%
  mutate(fitted_model = factor(SUBMODEL_MAP[fitted_model], levels = SUBMODEL_MAP))

# helper for multi-panel grids
theme_grid <- function(spacing_x = 1.4) {
  theme_cowplot(font_size = 11.5, font_family = "Tinos") +
    theme(
      legend.position = "top",
      panel.spacing.x = unit(spacing_x, "lines"),
      panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.5)
    )
}

# 3. PLOT BUILDERS
save_plot <- function(plot, filename, width, height) {
  ggsave(
    file.path(OUT_DIR, filename),
    plot   = plot,
    width  = width,
    height = height,
    dpi    = 150,
    device = ragg::agg_png
  )
}

plot_move_frequency <- function(trials_df) {
  trials_df %>%
    filter(!is.na(self_resp)) %>%
    mutate(
      move = factor(MOVE_LABELS[as.character(self_resp)], levels = MOVE_LABELS)
    ) %>%
    count(move) %>%
    mutate(proportion = n / sum(n)) %>%
    ggplot(aes(move, proportion, fill = move)) +
    geom_col(width = 0.65, show.legend = FALSE) +
    geom_hline(
      aes(yintercept = 1 / 3, linetype = "Uniform (1/3)"),
      linewidth = 0.6
    ) +
    scale_linetype_manual(values = "dashed") +
    scale_y_continuous(
      labels = label_percent(accuracy = 1),
      limits = c(0, NA),
      expand = expansion(mult = c(0, 0.1))
    ) +
    labs(title = "Move Frequency", x = NULL, y = "Proportion of Trials")
}

plot_pvalue_calibration <- function(participants_df) {
  ggplot(participants_df, aes(cramers_v, -log10(p_value))) +
    geom_point(alpha = 0.6, size = 1.8) +
    geom_vline(
      aes(xintercept = 0.25, linetype = "V = 0.25"),
      linewidth = 0.6
    ) +
    geom_hline(
      aes(yintercept = -log10(0.05), linetype = "p = .05"),
      linewidth = 0.6
    ) +
    scale_linetype_manual(values = c("V = 0.25" = "dotted", "p = .05" = "dashed")) +
    scale_x_continuous(labels = label_number(accuracy = 0.1)) +
    labs(
      title = "Significance vs. Effect Size (Calibration)",
      x = "Cramér's V",
      y = "−log₁₀(p-value)"
    )
}

plot_matched_density <- function(long_df) {
  long_df %>%
    pivot_longer(c(real, model), names_to = "source", values_to = "value") %>%
    mutate(source = recode(source, real = "Participant", model = "Model")) %>%
    ggplot(aes(value, color = source, fill = source)) +
    geom_density(alpha = 0.28, linewidth = 0.7) +
    geom_rug(alpha = 0.45, length = unit(0.035, "npc")) +
    facet_grid(stat_label ~ cohort_label, scales = "free") +
    labs(
      title = "Matched Cohort vs. Model Distributions",
      x = NULL,
      y = "Density"
    ) +
    theme_grid(spacing_x = 1.8)
}

plot_matched_forest <- function(long_df) {
  long_df %>%
    pivot_longer(c(real, model), names_to = "source", values_to = "val") %>%
    group_by(stat_label, cohort_label, source) %>%
    summarize(
      mean = mean(val, na.rm = TRUE),
      se   = sd(val, na.rm = TRUE) / sqrt(sum(!is.na(val))),
      .groups = "drop"
    ) %>%
    mutate(source = recode(source, real = "Participant", model = "Model")) %>%
    ggplot(aes(mean, cohort_label, color = source)) +
    geom_pointrange(
      aes(xmin = mean - se, xmax = mean + se),
      position = position_dodge(width = 0.45),
      size = 0.7
    ) +
    facet_wrap(~ stat_label, scales = "free_x", ncol = 4) +
    labs(
      title = "Matched Cohort vs. Model Summary (Mean ± 1 SE)",
      x = NULL,
      y = NULL
    ) +
    theme_grid(spacing_x = 1.8)
}

plot_bland_altman <- function(long_df, agreement_df, title) {
  faceted <- agreement_df %>%
    add_cohort_label() %>%
    add_stat_label() %>%
    mutate(label = sprintf("bias=%.2f\nWilcoxon p=%.3f", bias, wilcoxon_p))

  ggplot(long_df, aes((real + model) / 2, model - real)) +
    geom_point(alpha = 0.6, size = 1.8) +
    geom_hline(aes(yintercept = 0, color = "Zero bias"), linetype = "dotted") +
    geom_hline(data = faceted, aes(yintercept = bias, color = "Bias ± 95% LoA")) +
    geom_hline(
      data = faceted,
      aes(yintercept = limit_lower, color = "Bias ± 95% LoA"),
      linetype = "dashed"
    ) +
    geom_hline(
      data = faceted,
      aes(yintercept = limit_upper, color = "Bias ± 95% LoA"),
      linetype = "dashed"
    ) +
    geom_text(
      data = faceted,
      aes(x = -Inf, y = Inf, label = label),
      inherit.aes = FALSE,
      hjust = -0.05,
      vjust = 1.3,
      size = 2.6,
      lineheight = 0.9
    ) +
    facet_grid(stat_label ~ cohort_label, scales = "free") +
    scale_y_continuous(expand = expansion(mult = c(0.08, 0.25))) +
    labs(
      title = title,
      x = "Mean of Model & Participant",
      y = "Model − Participant"
    ) +
    theme_grid(spacing_x = 1.6)
}

plot_correlation <- function(long_df, title) {
  ggplot(long_df, aes(real, model)) +
    geom_abline(
      aes(slope = 1, intercept = 0, color = "y = x"),
      linetype = "dotted"
    ) +
    geom_point(alpha = 0.6, size = 1.8) +
    geom_smooth(
      aes(color = "Linear fit"),
      method = "lm",
      se = FALSE,
      linewidth = 0.7
    ) +
    stat_cor(
      method = "spearman",
      size = 2.6,
      r.accuracy = 0.01,
      p.accuracy = 0.001
    ) +
    facet_grid(stat_label ~ cohort_label, scales = "free") +
    scale_y_continuous(expand = expansion(mult = c(0.08, 0.22))) +
    labs(title = title, x = "Participant Value", y = "Model Value") +
    theme_grid(spacing_x = 1.8)
}

plot_submodel_counter <- function(wslc_df) {
  wslc_df %>%
    filter(stat == "counter_rate") %>%
    ggplot(aes(fitted_model, real, color = fitted_model)) +
    geom_boxplot(outlier.shape = NA, alpha = 0.25, show.legend = FALSE) +
    geom_jitter(width = 0.15, alpha = 0.75, size = 1.8, show.legend = FALSE) +
    geom_hline(
      aes(yintercept = 0.5, linetype = "Threshold (0.5)"),
      linewidth = 0.6
    ) +
    scale_linetype_manual(values = "dashed") +
    labs(
      title = "Counter Rate by Selected WSLC Sub-Model",
      x = NULL,
      y = "Counter Rate"
    )
}

plot_submodel_values <- function(wslc_df) {
  ggplot(
    wslc_df,
    aes(fitted_model, real, color = fitted_model, fill = fitted_model)
  ) +
    stat_dotsinterval(
      scale = 0.6,
      quantiles = NA,
      .width = 0.95,
      slab_color = NA,
      slab_alpha = 0.75,
      show.legend = FALSE
    ) +
    facet_wrap(~ stat_label, scales = "free_y", ncol = 4) +
    labs(
      title = "WSLC Participant Values by Selected Sub-Model",
      x = NULL,
      y = NULL
    ) +
    theme_grid(spacing_x = 1.2) +
    theme(axis.text.x = element_text(angle = 20, hjust = 1))
}

# 4. RENDER & SAVE
save_plot(
  plot_move_frequency(sheets$trials),
  "move_frequency.png",
  width = 6,
  height = 5
)
save_plot(
  plot_pvalue_calibration(sheets$participants),
  "pvalue_vs_cramers_v.png",
  width = 7,
  height = 5.5
)
save_plot(
  plot_matched_density(matched_long),
  "matched_density.png",
  width = 12,
  height = 10
)
save_plot(
  plot_matched_forest(matched_long),
  "matched_summary.png",
  width = 14,
  height = 4.5
)
save_plot(
  plot_submodel_counter(wslc_submodel_long),
  "counter_rate_by_submodel.png",
  width = 6,
  height = 5
)
save_plot(
  plot_submodel_values(wslc_submodel_long),
  "submodel_values.png",
  width = 14,
  height = 4
)
save_plot(
  plot_bland_altman(
    matched_long,
    sheets$agreement,
    "Model vs. Participant Agreement — Fixed Params"
  ),
  "bland_altman.png",
  width = 12,
  height = 10
)
save_plot(
  plot_correlation(
    matched_long,
    "Model vs. Participant Correlation — Fixed Params"
  ),
  "model_correlation.png",
  width = 12,
  height = 10
)
save_plot(
  plot_bland_altman(
    fitted_long,
    sheets$fitted_agreement,
    "Model vs. Participant Agreement — Per-Participant Fit"
  ),
  "bland_altman_fitted.png",
  width = 9,
  height = 10
)
save_plot(
  plot_correlation(
    fitted_long,
    "Model vs. Participant Correlation — Per-Participant Fit"
  ),
  "model_correlation_fitted.png",
  width = 9,
  height = 10
)