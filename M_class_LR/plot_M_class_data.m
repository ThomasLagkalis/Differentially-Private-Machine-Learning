function plot_M_class_data(X, y, M)
% plot_M_class_data(X, y, M)

if (M > 6), fprintf('\nNumber of classes larger than six. I return...'), return, end

[N, n] = size(X);
if (N ~= 2), return, end
hold on
for in = 1:n
    if ( y(in) == 1 )
        plot(X(1,in), X(2, in), '+b')
    elseif ( y(in) == 2 )
        plot(X(1,in), X(2, in), '+r')
    elseif ( y(in) == 3 )
        plot(X(1,in), X(2, in), '+c')
    elseif ( y(in) == 4 )
        plot(X(1,in), X(2, in), '+m')
    elseif ( y(in) == 5 )
        plot(X(1,in), X(2, in), '+y')
    elseif ( y(in) == 6 )
        plot(X(1,in), X(2, in), '+k')
    end
end
hold off