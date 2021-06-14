
# Random-Number-Generator
### RNG generating numbers from various distributions along with:
-  Series tests that check if numbers are generated randomly by constructing another series and comparing numbers of series in input with median value
- Chi-Square tests that check if generated numbers are of right distribution by comparing observed values with estimated values.
___

### There are seven generators implemented:

| Generator | Description |
|:-:|:-:|
| G | generates random integers |
| J | generates random fractions in range (0;1) |
| B | generates numbers with Bernoulli distribution |
| D | generates numbers with binomial distribution |
| P | generates numbers with Poisson distribution |
| W | generates numbers with exponential distribution |
| N | generates numbers with normal distribution |

Generator J is constructed using G, and B, D, P, W and N are constructed using J.
___
### Tests outcome:
I have tested every generator with series tests and the results were as expected: every generator except B and P generates correct pseudo-random numbers. Bernoulli cannot generate "very" random numbers due to the fact that is only generates '0' of '1'. Poisson on the other hand, has a lot duplicate values.
I have performed Chi-Square tests only for generators B and P. Results for every data set were satisfying.

| Test for 1000 generated numbers | G | J | B | D | P | W | N |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Series test | ✔️ | ✔️ | ❌ | ✔️ | ❌| ✔️ | ✔️ |
| Chi-Square | - | - | ✔️ | - | ✔️ | - | - |
<br>

| Test for 10 000 generated numbers | G | J | B | D | P | W | N |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Series test | ✔️ | ✔️ | ❌ | ✔️ | ❌| ✔️ | ✔️ |
| Chi-Square | - | - | ✔️ | - | ✔️ | - | - |
<br>

| Test for 50 000 generated numbers | G | J | B | D | P | W | N |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Series test | ✔️ | ✔️ | ❌ | ✔️ | ❌| ✔️ | ✔️ |
| Chi-Square | - | - | ✔️ | - | ✔️ | - | - |
<br>

| Test for 110 000 generated numbers | G | J | B | D | P | W | N |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Series test | ✔️ | ✔️ | ❌ | ✔️ | ❌| ✔️ | ✔️ |
| Chi-Square | - | - | ✔️ | - | ✔️ | - | - |
