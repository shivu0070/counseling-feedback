# Goal-setting / homework rubric (behavioral activation) and new feedback area

New taxonomy area: **Goal Setting** (added to Reflections, Questions, Suggestions, Validation, Structure, Empathy, Self-disclosure, Professionalism).
A response is "perfect" on Goal Setting when a task is set AND it meets all of items 1-6. Each controlled degradation in `goal_eval_set.json` breaks exactly one item.

| # | Item | Good looks like | Degradation (id) |
|---|------|-----------------|------------------|
| 1 | Specific & concrete | what + when + where ("Thursday evening, cook pasta for 30 min in your kitchen") | `vague` |
| 2 | Collaborative | invites the client's input / willingness ("Would you be open to...?") | `imposed` |
| 3 | Graded & achievable | small first step matched to current energy | `too_ambitious` |
| 4 | Linked to the client's values | ties the activity to something the client said matters/enjoyed | `no_value_link` |
| 5 | Anticipates barriers | asks what could get in the way and how to handle it | `no_barrier` |
| 6 | Plans a follow-up | agrees to review how it went next session | `no_followup` |

**Negative controls (`kind: "neg"`)**: client in acute distress (panic, fresh bereavement, assault disclosure, ...). The correct move is to be heard first.
`strong` = validating/exploring response with NO homework (should be "perfect"); `degraded` (`premature_homework`) = pushes a plan/homework (should be flagged).
A model that over-corrects toward homework will fail these.
