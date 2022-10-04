function f = sc_strip_map(input, index, interior)
     p = polygon(input);
     f = stripmap(p, index)
%      plot(f);
     interior_mapped = evalinv(f, interior)
     interior_final = eval(f, interior_mapped)
     scatter(real(interior_mapped), imag(interior_mapped))
     savefig('fullworm_strip_inv_dense.fig');
%     f.prevertex
%      plot(f.prevertex, 'o');
%      savefig('prevertex.fig');
end