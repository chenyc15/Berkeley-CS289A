function y_test = classifier(X_test, w)

X_test = [X_test, ones(size(X_test, 1), 1)];
y_test = 1./(1+exp(-X_test*w));
y_test = y_test>0.5;