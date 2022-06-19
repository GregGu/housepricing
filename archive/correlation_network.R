charvars <- names(df)[sapply(df, class) == 'character']
dftemp <- df %>%
  dplyr::mutate_at(charvars, as.factor) %>%
  dplyr::mutate(MSSubClass = as.factor(MSSubClass))


nom_vars <- names(dftemp)[sapply(dftemp, class) == 'factor']
num_ord_vars <- names(dftemp)[!names(dftemp) %in% nom_vars]

tempnom <- dftemp %>%
  dplyr::select(c("SalePrice", nom_vars)) %>% # including sale price with ordinal as well
  dplyr::select(-set) %>%
  mixed_assoc() %>%
  dplyr::select(x, y, assoc) %>%
  tidyr::spread(y, assoc) %>%
  tibble::column_to_rownames("x")

tempnum <- dftemp %>%
  dplyr::select(c("SalePrice", num_ord_vars)) %>% # including sale price with ordinal as well
  mixed_assoc() %>%
  dplyr::select(x, y, assoc) %>%
  tidyr::spread(y, assoc) %>%
  tibble::column_to_rownames("x")


temp <- dftemp %>%
  dplyr::select(c("SalePrice", num_ord_vars, nom_vars)) %>% # including sale price with ordinal as well
  dplyr::select(-set) %>%
  mixed_assoc() %>%
  dplyr::select(x, y, assoc) %>%
  tidyr::spread(y, assoc) %>%
  tibble::column_to_rownames("x")

# allowing for all overlap text
options(ggrepel.max.overlaps = Inf)

tempnom %>%
  as.matrix %>%
  corrr::as_cordf() %>%
  corrr::network_plot(min_cor = 0.3)
# corrplot::corrplot(tempnom %>% as.matrix %>% .[dim(tempnom)[1]:1,dim(tempnom)[1]:1], method = 'square', order = 'FPC', type = 'lower', diag = TRUE, tl.col = "black", tl.cex=.75)

tempnum %>%
  as.matrix %>%
  corrr::as_cordf() %>%
  corrr::network_plot(min_cor = 0.3)
