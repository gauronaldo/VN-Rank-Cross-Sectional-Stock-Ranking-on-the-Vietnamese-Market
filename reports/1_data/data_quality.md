# Data-quality report

## Snapshot

- **completed_at:** 2026-11-07T09:47:03+00:00
- **empty_symbols:** ['HLE', 'KSS', 'MEG', 'PYU', 'SON', 'SVL', 'TAC', 'UTT', 'VDM', 'VIS']
- **end:** 2026-11-06
- **exchanges:** ['HOSE', 'HNX', 'UPCOM', 'DELISTED']
- **failed_symbols:** ['BLU', 'BPW', 'BUD', 'BXT', 'CT5', 'DKH', 'DNB', 'DX2', 'GQN', 'GTK', 'HAB', 'HAW', 'HGA', 'HGC', 'HGR', 'HNE', 'HTK', 'HUX', 'IPH', 'MXC', 'NNQ', 'PDT', 'T12', 'TS5', 'VHI']
- **index_symbol:** VNINDEX
- **last_run_vnstock_version:** 5.2.2
- **source:** vci
- **start:** 2018-11-01
- **started_at:** 2026-11-07T09:10:42+00:00
- **status_counts:** {'empty': 10, 'failed': 25, 'ok': 1720}
- **vnstock_version:** 5.2.2

## Flags per check

| index | rows | symbols |
|---|---|---|
| duplicate | 0 | 0 |
| missing_value | 0 | 0 |
| non_positive_price | 16 | 16 |
| ohlc_inconsistent | 67391 | 841 |
| zero_volume | 1009415 | 1411 |
| band_breach | 18496 | 1092 |
| off_calendar | 0 | 0 |

## Coverage

1720 symbols with data; median rows 1912; symbols with at least one missing VNINDEX session: 1195.

| symbol | first | last | rows | traded_days | expected_sessions | missing_sessions | share |
|---|---|---|---|---|---|---|---|
| DKW | 2025-02-11 | 2025-12-10 | 5 | 5 | 211 | 206 | 0.9763 |
| UXC | 2025-01-24 | 2026-09-11 | 30 | 30 | 402 | 372 | 0.9254 |
| PTM | 2018-11-05 | 2026-11-02 | 228 | 144 | 1994 | 1766 | 0.8857 |
| SLD | 2025-08-15 | 2026-09-17 | 60 | 60 | 270 | 210 | 0.7778 |
| ING | 2024-05-15 | 2025-10-10 | 80 | 36 | 354 | 274 | 0.7740 |
| XDC | 2022-12-01 | 2023-12-28 | 68 | 68 | 270 | 202 | 0.7481 |
| DKG | 2025-04-22 | 2026-10-12 | 101 | 101 | 367 | 266 | 0.7248 |
| UPS | 2026-03-20 | 2026-10-30 | 44 | 44 | 155 | 111 | 0.7161 |
| DVT | 2025-07-18 | 2026-03-18 | 53 | 53 | 165 | 112 | 0.6788 |
| TAB | 2024-09-27 | 2026-11-02 | 170 | 170 | 522 | 352 | 0.6743 |
| GTX | 2026-09-11 | 2026-11-06 | 16 | 16 | 41 | 25 | 0.6098 |
| NEM | 2024-01-05 | 2025-09-15 | 167 | 116 | 419 | 252 | 0.6014 |
| CK8 | 2023-01-06 | 2026-10-23 | 393 | 10 | 944 | 551 | 0.5837 |
| ANI | 2026-07-01 | 2026-09-03 | 19 | 19 | 44 | 25 | 0.5682 |
| BGE | 2024-07-31 | 2026-11-06 | 277 | 277 | 566 | 289 | 0.5106 |

## Symbols traded per year

| year | listed | delisted |
|---|---|---|
| 2018 | 1135 | 119 |
| 2019 | 1263 | 141 |
| 2020 | 1312 | 145 |
| 2021 | 1373 | 157 |
| 2022 | 1404 | 163 |
| 2023 | 1412 | 144 |
| 2024 | 1423 | 95 |
| 2025 | 1440 | 71 |
| 2026 | 1456 | 33 |

## Largest band breaches (top 15)

| date | symbol | exchange | ret |
|---|---|---|---|
| 2021-08-03 | DNN | UPCOM | 32.4987 |
| 2021-02-18 | DNN | UPCOM | 32.4987 |
| 2021-01-25 | DNN | UPCOM | 32.4987 |
| 2021-01-08 | DNN | UPCOM | 32.4987 |
| 2021-08-23 | DNN | UPCOM | 32.4987 |
| 2021-09-06 | DNN | UPCOM | 32.4987 |
| 2020-11-19 | PTG | UPCOM | 20.0084 |
| 2021-01-25 | PTG | UPCOM | 20.0000 |
| 2021-02-18 | PTG | UPCOM | 20.0000 |
| 2021-01-08 | PTG | UPCOM | 20.0000 |
| 2025-08-22 | PTM | UPCOM | 7.0000 |
| 2021-05-20 | HHR | DELISTED | 5.4020 |
| 2021-01-25 | PTX | HNX | 5.0001 |
| 2021-01-08 | PTX | HNX | 5.0001 |
| 2021-08-23 | PTX | HNX | 5.0001 |
