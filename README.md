### SBML File Generator Wizard

The wizard has 2 different modes.

1. Questions about specifications

Uses a local Ollama model (currently, Gemma3:4b) with a RAG pipeline (cosine similarity) to inject knowledge about SBML Multi from specification into model for user to query. 

2. SBML generation

Uses a cloud Ollama model (currently, minimax-m3) to generate an SBML file from an image provided. Users can upload their own image and save the results to disk.
