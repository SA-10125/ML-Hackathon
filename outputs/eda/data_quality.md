# Data-quality record (train + student_test)

| Check | Result | Decision |
|---|---|---|
| Missing / NaN GPS values | 0 rows | none to handle |
| Coordinates out of range | 0 rows | none to handle |
| Duplicate or unsorted timestamps | 0 rows | sort by time, drop exact duplicates (in loader) |
| Steps exactly 1 s apart | train 96.5% | nominal 1 Hz confirmed |
| Trips with a gap > 10 min | 6: 00000090 (42.5 h), 00000078 (14.7 h), 00000069 (0.7 h), 00000021 (0.6 h), 00000011 (0.3 h), 00000042 (0.2 h) | dt features capped at 600 s; HMM must break/relax transitions across gaps |
| Implied speed > 180 km/h | 1538 steps | GPS jumps; kept (no ground truth to delete them), HMM transitions penalise them |
| Reference route does not cover the GPS track | 00000054 (89% of fixes), 00000033 (59% of fixes), 00000002 (12% of fixes), 00000066 (9% of fixes), 00000043 (7% of fixes) | fixes > 50 m from their aligned reference arc labelled unreliable (-1), excluded from training and scoring |
| Reference route not fully connected | 11 trips (min continuity 0.981) | treated as reference with known gaps, not error-free truth |
| Isolated nodes | 0 across all trips | ignored (no arcs, never candidates) |
| Directed arcs with a reverse twin | median 95% | heading feature needed to separate A->B from B->A |
| Weakly connected components | median 19 per trip | transitions between components are impossible -> HMM must handle |
| Candidate radius | 40 m: miss rate 0.55%, 7.7 candidates/point | chosen R = 40 m (train only). Note: miss rate is 0 for R >= 50 m by construction, because labels > 50 m are excluded, so only R < 50 m is informative |

Trips span several continents, so distances use a per-trip UTM projection (metres), never raw degrees. Coordinates are stored lon, lat.
