Here is the next sequential set of highly structured, iterative prompts designed to take your foundational prototype and elevate it into a live, end-to-end dynamic system. These prompts guide VSCode Copilot to implement real-time data streaming, stateful agentic orchestration, live UI/dashboarding, and concurrent hardware management, all while strictly adhering to your 8GB VRAM hardware constraints.

Submit these sequentially to your VSCode Copilot agent:

### **Phase 8: Real-Time Event Streaming and Change Data Capture (CDC)**

**Rationale**: AI agents make catastrophic decisions if they act on stale data. To make the system truly live, we must replace simulated asynchronous loops with an event-driven architecture that captures database changes in milliseconds.

**Prompt 13: Live Event-Driven Streaming Architecture**

"We are moving our fraud analyzer prototype into a live, dynamic state. The AI agents must not rely on batch ETL or stale data.

Task: Implement a real-time event streaming pipeline utilizing the most robust Change Data Capture (CDC) and event-streaming libraries available for Python in 2026\. Do not hardcode specific platform names; choose the best memory-efficient tools. Write the integration code that listens to live database inserts (simulated live banking endpoints) and streams these state changes instantly to our previously built Machine Learning and Graph Analytics ingestion layers. Ensure the streaming consumer is heavily optimized to run efficiently as a background process on a Windows environment without monopolizing CPU threads."

### **Phase 9: Dynamic Agentic Orchestration and State Management**

**Rationale**: Deterministic, rule-based workflows fail against sophisticated, evolving fraud. The Qwen 3.5 9B orchestrator must transition from a simple prompt-response loop to a stateful, dynamic workflow with built-in memory and escalation protocols.

**Prompt 14: Stateful Agent Orchestration and Human-in-the-Loop (HITL)**

"Our Qwen 3.5 9B orchestrator needs to handle unpredictable fraud paths dynamically rather than following a fixed decision tree.

Task: Implement an advanced, state-machine-based agentic orchestration framework. Write the logic that allows the Investigator Agent to iteratively plan, execute tool calls, evaluate intermediate findings, and decide the next best action at runtime. Additionally, implement a strict 'Human-in-the-Loop' (HITL) escalation protocol. If the agent's confidence score regarding a complex money laundering typology falls below a programmatic threshold, the system must securely halt the autonomous loop, package the graph context, and route a summarized alert to a human analyst API endpoint."

### **Phase 10: Live Blockchain Circuit Breakers**

**Rationale**: Detecting fraud is only half the battle; the system must autonomously intercept it. This requires deploying the previously built smart contracts as active circuit breakers listening to the live data stream.

**Prompt 15: Automated Ecosystem Controls and Circuit Execution**

"We must operationalize our local blockchain to act as an active defense mechanism for high-volume transactions.

Task: Write the execution listeners that bind our stateful AI agent's outputs to the local smart contract circuit breakers. If the agentic workflow issues a 'CRITICAL\_FRAUD\_DETECTED' state, implement the deterministic code that triggers the blockchain to instantaneously simulate freezing the compromised nodes, pausing routing, and rejecting further incoming transactions from the associated device fingerprints. Ensure this circuit-breaking execution happens in milliseconds and logs the exact cryptographic proof of the AI's reasoning into the immutable ledger for regulatory compliance."

### **Phase 11: Real-Time Graph Visualization Dashboard**

**Rationale**: A live intelligence model requires a dynamic frontend for human analysts to monitor the AI's autonomous actions, view the multi-hop network topologies, and interact with the data.

**Prompt 16: Interactive AI-Driven Analytics Frontend**

"The intelligence model requires a live, visual frontend to monitor the graph topology and AI agent states.

Task: Generate a lightweight, real-time web dashboard using a modern, reactive Python web framework and a highly performant, embeddable graph visualization library. The dashboard must consume the live event streams to dynamically update nodes and edges on the screen as transactions occur. Create specific UI components that display the Qwen 3.5 9B agent's 'Chain-of-Thought' log in real-time, the live probabilistic risk scores, and the current status of the blockchain circuit breakers. Keep the frontend dependencies extremely light so rendering does not compete for the system's limited resources."

### **Phase 12: Production-Grade VRAM Concurrency Management**

**Rationale**: A live environment experiences bursty, unpredictable traffic spikes. The system will inevitably crash the RTX 4070's 8GB VRAM if the LLM, ML models, and Graph updates attempt to process massive concurrent spikes without strict queuing.

**Prompt 17: Concurrency Queuing and GPU Residency Management**

"The prototype is now end-to-end and dynamic, meaning it will face unpredictable, bursty traffic patterns. We must absolutely protect the 8GB VRAM limit of the RTX 4070 during concurrent execution.

Task: Refactor the main application loop to implement an advanced concurrency management system. Write a priority queue that dictates GPU access. The Qwen 3.5 9B model must remain resident in VRAM to prevent unacceptable cold-start latency. Implement dynamic sequence limiting (max concurrent sequences) to strictly cap the LLM's KV cache growth during traffic spikes. If the VRAM approaches 7.8GB, the system must instantly queue incoming graph embedding updates to the CPU's system RAM, sacrificing analytical speed temporarily to preserve the operational stability of the live agentic decision engine."

