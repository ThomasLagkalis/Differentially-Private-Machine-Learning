# Private (Noisy) Average

Algorithms implemented:

1. **Simple Noisy Average**: The simplest version of ϵ-DP average, in which we just add noise to the summation over the dataset D and to the count of D.
2. **Noisy Average with Resampling:** Average is computed without adding noise to the count function. This algorithm does not satisfy ϵ-DP.
3. **Noisy Average with Clamping-Down:**If the noisy average falls outside the desired range, then it is clamped down instead of resampled. In this way, the algorithm is ϵ-DP.
4. **Noisy Average with Normalization:** We add noise to both sum and count functions but using a normalization trick to reduce the sensitivity of the sum query.

Check the [thesis report](../Thesis_report.pdf) for details.
