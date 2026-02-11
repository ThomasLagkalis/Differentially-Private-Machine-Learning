function []= plot_decision_regions(W, x1, x2)
% plot_decision_regions(W, x1, x2)

hold on
for ix1 = 1:length(x1)
    for ix2 = 1:length(x2)
            tmp_x = [x1(ix1); x2(ix2)];
            [~, pos] = max(W' * tmp_x);
            if (pos == 1)
                plot(x1(ix1), x2(ix2), 'ob')
            elseif (pos == 2)
                plot(x1(ix1), x2(ix2), 'or')
            elseif (pos == 3)
                plot(x1(ix1), x2(ix2), 'oc')
            elseif (pos == 4)
                plot(x1(ix1), x2(ix2), 'om')
            elseif (pos == 5)
                plot(x1(ix1), x2(ix2), 'oy')
            elseif (pos == 6)
                plot(x1(ix1), x2(ix2), 'ok')   
            end
    end
end
hold off