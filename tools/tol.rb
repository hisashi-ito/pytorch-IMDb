#! /usr/bin/ruby
cnt = 0
while line = gets
  line = line.chomp.encode("utf-8")
  if line =~ (/loss/)
    elems = line.split("\s")
    loss = elems[8].to_f
    train_acc = elems[10].to_f
    val_acc = elems[12].to_f
    cnt += 1
    puts("#{cnt}\t#{loss}\t#{train_acc}\t#{val_acc}")
  end
end
