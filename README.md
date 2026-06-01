#  OS Banker's Algorithm Simulation

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An interactive, animated dashboard designed to visualize deadlock avoidance in Operating Systems. Built with Python and Streamlit, this simulation takes the Banker's Algorithm out of the textbook by generating dynamic Resource Allocation Graphs (RAG) and animating the safety sequence step-by-step.

##  Key Features

- **Dynamic RAG Generation:** Automatically builds and updates Resource Allocation Graphs using Graphviz to visually map resource claims, allocations, and requests.
- **Step-by-Step Animation:** Watch the algorithm evaluate the system state in real-time. The UI color-codes processes as they are evaluated (Processing), approved (Finished), or halted (Blocked).
- **Hypothetical Request Testing:** Inject manual resource requests on the fly. The engine simulates the allocation *before* granting it, rejecting unsafe requests and preventing system deadlocks.
- **Custom System Configurations:** Easily scale the simulation from simple 3-process models to complex 20-process systems with up to 10 distinct resource types.

##  How the Simulation Works

The engine strictly follows Dijkstra's Banker's Algorithm for deadlock avoidance:
1. **State Initialization:** Calculates the `Remaining Need` matrix (`Max Need` - `Allocation`).
2. **Safety Check:** Iterates through all processes to find a safe execution sequence where every process's remaining needs can be met by the `Available` resources pool.
3. **Resource Granting:** If a manual request is made, the engine temporarily allocates the resources, runs the safety algorithm, and only commits the state change if the resulting state is mathematically safe.

##  Installation & Setup
```bash
  #Clone the repository:
  git clone [https://github.com/princySpatel/banker-s-algorithm-demonstration-.git](https://github.com/princySpatel/banker-s-algorithm-demonstration-.git)
  cd banker-s-algorithm-demonstration-
```
```bash
  #Install the required dependencies:
  #This project relies on the standard data science stack and Streamlit for the UI.
  pip install streamlit numpy pandas graphviz
```
```bash
  #Run the application:
  streamlit run app.py
```

##  Usage Guide

1. Use the setup inputs to define the **Number of Processes** and **Number of Resources**.
2. Fill in the **Max Need Matrix** and **Allocation Matrix** for each process.
3. Input the initial **Available Resources**.
4. Click **Show Current State Graph** to generate the initial RAG.
5. Click **Check Safety** to watch the algorithm step through the execution sequence and determine if the system is safe.
6. Use the **Request Resource** section to simulate a process asking for specific resources dynamically.

##  License

This project is open-source and available under the MIT License.
