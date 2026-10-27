## Mesh refinement OCP

Approaches to refine mesh in OCP

1. static - solve nlp -> compute error -> add/delete nodes based on metric -> solve nlp (warm start) -> repeat till convergence 
2. dynamic - discrete time/ time segment width/ fraction treated as decision variables -> solve single NLP but much more complex than original OCP

