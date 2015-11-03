function [output] = EMTest()

u1 = 152;
o1 = 9;
u2 = 178;
o2 = 10;

girls = randn(100,1) * o1 + u1;
boys = randn(100,1) * o2 + u2;
combined = [girls;boys];

[bins, centers] = hist(combined);
binSum = sum(bins) * (centers(2) - centers(1));
bar(centers, bins/binSum);

meanData = mean(combined);

standardDeviation = std(combined);
%theoVariance = 0.5 * (o1^2) + 0.5 * (o2^2) + ((0.5 * (u1^2) + 0.5 * (u2^2)) - (0.5 * u1 + 0.5 * u2)^2);
%theoVariance = sqrt(theoVariance);
theoVar = theoVariance(0.5,o1,o2,u1,u2);

hold on
X = centers(1):centers(end);
plot(X, normpdf(X, meanData, standardDeviation))
plot(X, normpdf(X, meanData, theoVar))

[meanEM, stdEM, pgEM] = EM(combined, 2)

a = (normpdf(X, meanEM(1), stdEM(1)) + normpdf(X, meanEM(2), stdEM(2))) * 0.5;
plot(X, a)

%plot(X, normpdf(X, meanEM(1), stdEM(1)))
%plot(X, normpdf(X, meanEM(2), stdEM(2)))

theoVar2 = theoVariance(0.5,stdEM(1),stdEM(2),meanEM(1),meanEM(2));
plot(X, normpdf(X, meanEM(1), theoVar2))
plot(X, normpdf(X, meanEM(2), theoVar2))

end

function [theoVar] = theoVariance(p,o1,o2,u1,u2)
theoVar = p * (o1^2) + p * (o2^2) + ((p * (u1^2) + p * (u2^2)) - (p * u1 + p * u2)^2);
theoVar = sqrt(theoVar);
end

