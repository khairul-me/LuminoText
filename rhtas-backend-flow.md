graph TD
    subgraph Initialize["1. System Initialization"]
        A1[Load Required Models]-->A2[Initialize NLP Pipeline]
        A2-->A3[Setup T5 Transformer]
        A3-->A4[Configure VADER Sentiment]
    end

    subgraph TextProcess["2. Text Processing"]
        B1[Extract Text from PDF]-->B2[Clean Text]
        B2-->B3[Remove Special Characters]
        B3-->B4[Normalize Text]
        B4-->B5[Tokenization]
    end

    subgraph Analysis["3. Parallel Analysis Pipeline"]
        C1[Preprocessed Text] --> D1[Topic Modeling]
        C1 --> D2[Sentiment Analysis]
        C1 --> D3[Timeline Analysis]
        C1 --> D4[Network Analysis]
        C1 --> D5[Keyword Analysis]

        D1 -->|LDA Algorithm| E1[Topic Distribution]
        D2 -->|VADER| E2[Sentiment Scores]
        D3 -->|NER + Date Extraction| E3[Chronological Events]
        D4 -->|Graph Construction| E4[Concept Relations]
        D5 -->|TF-IDF| E5[Keyword Frequencies]
    end

    subgraph Visualize["4. Visualization Generation"]
        F1[Generate Topic Clouds]
        F2[Create Sentiment Charts]
        F3[Plot Timeline]
        F4[Draw Network Graph]
        F5[Plot Frequency Charts]
    end

    subgraph Report["5. Report Generation"]
        G1[Compile Analysis Results]
        G2[Format Visualizations]
        G3[Generate Summary]
        G4[Create Final Report]
    end

    %% Connections between subgraphs
    Initialize -->|Load Complete| TextProcess
    TextProcess -->|Cleaned Text| Analysis
    Analysis -->|Analysis Results| Visualize
    Visualize -->|Visual Elements| Report

    %% Styling
    classDef process fill:#e6f3ff,stroke:#4a90e2,stroke-width:2px
    classDef analysis fill:#e8f5e9,stroke:#4caf50,stroke-width:2px
    classDef visual fill:#fce4ec,stroke:#e91e63,stroke-width:2px
    classDef report fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px

    class A1,A2,A3,A4 process
    class B1,B2,B3,B4,B5 process
    class C1,D1,D2,D3,D4,D5,E1,E2,E3,E4,E5 analysis
    class F1,F2,F3,F4,F5 visual
    class G1,G2,G3,G4 report
