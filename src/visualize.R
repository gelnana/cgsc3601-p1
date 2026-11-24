library(readxl)
library(dplyr)
library(ggplot2)

participants <- read_excel("data/clusters.xlsx", sheet = "participants")
comparison <- read_excel("data/clusters.xlsx", sheet = "comparison") %>%
  mutate(cohort = factor(cohort, levels = c("none", "WCLS", "WSLC-C", "WSLC-I")))
trials <- read_excel("data/clusters.xlsx", sheet = "trials")

theme_set(theme_minimal(base_size = 13))

# --- move frequency ---

move_labels <- c(`1` = "Rock", `2` = "Paper", `3` = "Scissors")

move_proportions <- trials %>%
  filter(!is.na(self_resp)) %>%
  mutate(move = factor(move_labels[as.character(self_resp)], levels = move_labels)) %>%
  count(move) %>%
  mutate(proportion = n / sum(n))

move_plot <- ggplot(move_proportions, aes(x = move, y = proportion, fill = move)) +
  geom_col() +
  geom_hline(yintercept = 1 / 3, linetype = "dashed", color = "grey40") +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "Move frequency across all recorded trials",
    subtitle = "dashed line: expected under uniform random play",
    x = NULL,
    y = "proportion of trials"
  ) +
  theme(legend.position = "none")

ggsave("data/move_frequency.png", move_plot, width = 6, height = 5, dpi = 150)

