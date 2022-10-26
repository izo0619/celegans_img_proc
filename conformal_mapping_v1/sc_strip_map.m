function [interior_mapped, colors] = sc_strip_map(input, index, interior, colors)
     p = polygon(input);
     f = stripmap(p, index);
     interior_mapped = evalinv(f, interior);
     interior_final = eval(f, interior_mapped);
     indices = find(~isnan(interior_final));
     colors = colors(indices);
     interior_mapped = interior_mapped(indices);
     ratio = length(indices)/length(interior)
%      interior_mapped = interior_mapped(indices);
%      figure
%      scatter(real(interior_mapped), imag(interior_mapped));
%      savefig('interior_canonical.fig');
%      hold on
%      
%      hold on
%      scatter(real(interior), imag(interior), 'filled');
%      plot(input, LineStyle="-", Color='green');
%      scatter(real(interior(indices)), imag(interior(indices)), MarkerFaceColor='red');
%      savefig('bounded_input.fig');
end