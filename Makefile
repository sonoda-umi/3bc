# ----------------------------------------------------------------              
# SLACK AUTOMATION - DEVELOPMENT TASKS                                          
# ----------------------------------------------------------------              

.PHONY: sort check format run test type-check lint                             

# Sort imports using Ruff                                                       
sort:                                                                           
	uv run ruff check --select I --fix .                                        

# Check formatting without making changes                                       
check:                                                                          
	uv run ruff format --check .                                                

# Auto-format the entire codebase                                               
format:                                                                         
	uv run ruff format .                                                        

# Run type checking (Strict mode)                                                                     
type-check:                                                                     
	uv run pyright                                                              

# Run all linting tasks (Sorting, Formatting check, and Type checking)          
lint: sort format type-check                                                    
