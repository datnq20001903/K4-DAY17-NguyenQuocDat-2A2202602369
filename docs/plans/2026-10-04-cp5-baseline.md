# CP5: honest baseline agent

Keep `SessionState` fields and key sessions exclusively by `thread_id`.
The baseline retains all conversation messages in RAM, reads no profiles,
and keeps `compaction_count()` equal to zero.

1. Add failing tests for same-thread recall, cross-thread and cross-instance
   isolation, no profile IO, full history retention, and cumulative accounting.
2. Implement deterministic offline recall by extracting facts from user messages
   in the current thread only. Return `answer`, `agent_tokens`, `prompt_tokens`
   for each turn. Count prompt context before appending the new assistant reply.
3. Implement optional LangChain construction through `build_chat_model()`.
   Pass the complete local session as live input; use no checkpointer, tools,
   persistent store or compact middleware. Commit successful turns only.
4. Verify live routing with a local fake chat model and the real LangChain graph.
   Record actual output-token metadata when available, otherwise estimate it;
   prompt accounting remains the same text estimate as offline mode.
5. Run CP3/CP4 regressions, demonstrate two threads for one user, review changes,
   and document the protocol and limitations in `Analysis.md`.

Advanced Agent and the two scaffold tests that depend on it remain later work.
