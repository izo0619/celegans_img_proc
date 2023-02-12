function f = sc_rect_map(input, index, interior)
     p = polygon(input);
     f = rectmap(p, index);
     interior_mapped = evalinv(f, interior)
     interior_final = eval(f, interior_mapped)
     scatter(real(interior_mapped), imag(interior_mapped))
     savefig('test.fig');
     plot(f);
     savefig('rectmap_56-1-25-27-54.fig')
end