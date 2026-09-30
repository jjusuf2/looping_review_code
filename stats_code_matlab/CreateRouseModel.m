function rm = CreateRouseModel(tau, deltaS, D, gamma, kappa, sigma_loc)



rm.acfRouse = @(tau) D*(sqrt(gamma*tau/(4*pi*kappa)).*(exp(-gamma*deltaS^2./(4*kappa*tau)) - 1) ...
    + gamma*deltaS/(4*kappa)*erf(sqrt(gamma*deltaS^2./(4*kappa*tau))));

rm.deltaS = deltaS;
rm.D = D;
rm.gamma = gamma;
rm.kappa = kappa;
rm.tau = tau;
rm.sigma_loc = sigma_loc;

N = length(tau);
dt = tau(2)-tau(1);
rm.Sigma = zeros(N, N);

for i=1:N
    for j=1:N
        rm.Sigma(i, j) = rm.acfRouse(abs(j-i)*dt);
    end
end



