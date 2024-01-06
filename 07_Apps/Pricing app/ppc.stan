data {
    int<lower=0> N;  // Number of observations
    int<lower=1> J;  // Number of potential outcomes
    array[N] real price;  // Regressor (price)
}

parameters {
    array[J] real alpha;  // Intercepts for each outcome
    real<upper=0> beta;  // Single coefficient for price, constrained to be negative
    real mu_alpha;  // Global mean for alpha distribution
    real<lower=0> sigma_alpha;  // Global standard deviation for alpha distribution
}

model {
    // Priors for hyperparameters
    mu_alpha ~ normal(0, 5);
    sigma_alpha ~ inv_gamma(2, 0.5);

    // Priors for varying intercepts
    alpha ~ normal(mu_alpha, sigma_alpha);

    // Single beta prior
    beta ~ normal(0, 5);  // Adjust the prior as needed

}

generated quantities {
    matrix[N, J] choice_probabilities;  // Probabilities for each choice over all observations

    for (n in 1:N) {
        vector[J] utilities;
        for (j in 1:J) {
            utilities[j] = alpha[j] + beta * price[n];  // Use the single beta
        }
        choice_probabilities[n] = to_row_vector(softmax(utilities));
    }
}
