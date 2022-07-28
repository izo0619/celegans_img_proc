function f = sc_mapping(input)
     p = polygon(input);
%      p=polygon([i -1+i -1-i 1-i 1 0]);
     plot(p)
     f = diskmap(p);
     f = center(f,-0.5-0.5i);
     plot(f);
end