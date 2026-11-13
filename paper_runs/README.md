# Paper runs

These scripts produced every number, table and figure of the paper. Their results are saved here, so nothing
needs to be rerun to inspect or rebuild the tables. Run each command from inside the folder named.

## farm/ (Section 4 and Appendix C: the farming community)
```
python farm_run.py plans            # 11 plans x 5 measurement durations x 40 communities      (~6 min)
python farm_run.py checks           # placebo, lag scan, DMDc / output Granger / Lasso-Granger  (~5 min)
python farm_run.py channels         # channel certificates from pilots of 1-5 years             (~25 min)
python farm_run.py wrong_lag        # pest -> spraying at a wrong lag vs the lag scan           (~1 min)
python farm_run.py physical         # local aquifer: social and physical networks, lags from the  (~1 min)
                                    #   sketch of each variant (with seepage, pumping is not tested)
python farm_run.py hidden           # two farms are not measured                                (~1 min)
python farm_run.py ours_base_data   # our test on the baselines' data (Table 2, Figure 3)       (~1 min)
python farm_run.py ours_daily_outputs  # the same without the weekly survey (Section 4)       (~1 min)
python farm_run.py cgranger         # conditional Granger baseline                              (~2 min)
python farm_run.py pool_corr        # correlation of the pooled residuals                       (~1 min)
python farm_analyze.py              # -> results/farm_summary.json
python farm_tables.py               # -> tables/tab_base.tex (Table 2), tab_plans.tex (Table 3),
                                    #    tab_farmcert.tex (Table 4), numbers.md (numbers in the text)
python fig_farm.py                  # -> figures/fig_farm.pdf (Figure 1)
python fig_network.py               # -> figures/fig_network.pdf (Figure 3)
```
`farm_run.py <part>` writes `results/farm_<part>.pkl`. Community s is built with seed s and simulated with seed
1000 + s, so every run is deterministic; `../verify.py` reruns community 0 of every part and checks that the
result is identical to the saved one. The model, the test and the plans are in `../lib/`.

Table 2 compares all methods on the same simulated data (the 5-year runs of `checks`, `cgranger` and
`ours_base_data`). Table 3 uses the 10-year runs of `plans`; its first 1-5 years are a different draw of the same
40 communities, so the two tables agree to rounding but are not the same runs.

## generic/ (Appendix D and Table 5: the generic benchmark)
```
python run_paper.py E1       # validity and power vs measurement duration, baselines      (Table 5, row 1)
python run_paper.py E1b      # heavy tails, shared drivers, unmeasured drivers, heterogeneity  (rows 2-4)
python run_paper.py S1       # contacts per agent                                          (row 5)
python run_paper.py S2       # ring with links moved to non-neighbours                     (row 6)
python run_paper.py S3       # channels per agent                                          (row 7)
python run_paper.py S4       # network size up to 500 agents                               (row 8)
python run_paper.py E5       # aggregation over Delta periods                              (row 10)
python run_extra.py L        # lag too short / true / too long, and the lag scan           (row 9)
python analyze_paper.py; python analyze_extra.py   # -> paper_summary.json, extra_summary.json
python bench_table.py        # -> tables/tab_bench.tex (Table 5)
```
`run_paper.py E3`, `E4` and `B` (a shared input only, measurement plans, blindness checks) and `run_extra.py LG`
(Lasso-Granger) are further checks that are not reported in Table 5. Set `SMOKE=1` for a quick small run of
`run_paper.py`. The raw `.pkl` results of the generic benchmark (about 120 MB, mostly S4) are not included, only
the summaries; the analysis scripts update the sections whose `.pkl` files are present and keep the others.
E1 rerun from scratch reproduces the saved summary exactly (about 2 minutes).
