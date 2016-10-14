sed 's/$/.png/' X_small_train.txt > tmpfileSTr.txt # add .bytes to tail of each line, and save the tmp file
sed -e 's#^#https://s3.amazonaws.com/eds-uga-csci8360/data/project3/images/#' tmpfileSTr.txt > trainSmallPath.txt # add 'http' address to head of each line
sed 's/$/.png/' X_small_test.txt > tmpfileSTe.txt # add .bytes to tail of each line, and save the tmp file
sed -e 's#^#https://s3.amazonaws.com/eds-uga-csci8360/data/project3/images/#' tmpfileSTe.txt > testSmallPath.txt # add 'http' address to head of each line
sed 's/$/.png/' X_train.txt > tmpfileTr.txt # add .bytes to tail of each line, and save the tmp file
sed -e 's#^#https://s3.amazonaws.com/eds-uga-csci8360/data/project3/images/#' tmpfileTr.txt > trainPath.txt # add 'http' address to head of each line
sed 's/$/.png/' X_test.txt > tmpfileTe.txt # add .bytes to tail of each line, and save the tmp file
sed -e 's#^#https://s3.amazonaws.com/eds-uga-csci8360/data/project3/images/#' tmpfileTe.txt > testPath.txt # add 'http' address to head of each line