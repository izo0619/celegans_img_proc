function f = sc_strip_map(input, index)
     p = polygon(input);
     f = stripmap(p, index);
     f = center(f,-0.5-0.5i);
     plot(f);
end