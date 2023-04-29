function [Gbest,FE1,PF_Time]=cosEMT
% num00=cell(num_auxiliary+1,1);
Gbest=[];
FE1=[];
PF_Time=[];
% for time=1:10
open=1;
XX = data_generate(open);
[kk,~]=size(XX);
indices=crossvalind('Kfold',XX(1:kk,1),5);
i=1;
test = (indices ==i); %获得test集元素在数据集中对应的单元编号
train = ~test;%train集元素的编号为非test元素的编号
train_data=XX(train,:);%从数据集中划分出train样本的数据
test_data=XX(test,:);%test样本集
X_train=train_data;
Y_test=test_data;
X0=X_train;
Y=Y_test;
Num_row=size(X0,1);
Num_column=size(X0,2);
t=clock;%潮流计算开始时间
CR=0.8;
MR=0.1;
u=10;
N=100;
num_auxiliary=3;
maxFE=4000;
length1=[];
length2=[];
length3=[];
L1=cell(num_auxiliary,1); L2=cell(num_auxiliary,1);
for j=1:num_auxiliary+1
    L1{j}=25;
    L2{j}=25;
end
pop0=cell(num_auxiliary+1,1);
for i=1: num_auxiliary+1
    pop0{i}=initialize(X0,N/(num_auxiliary+1));
end
% dim=size(pop0{1}(1,:),2);
fit0=cell(num_auxiliary+1,1);
acc0=cell(num_auxiliary+1,1);
for i=1:num_auxiliary+1
    num=size(pop0{i},1);
    for j=1:num
        [fit0{i}(j,:),acc0{i}(j,:)]=FitnessFunction(X0,Y,pop0{i}(j,:));
    end
end
latency0=cell(num_auxiliary+1,1);
for i=1:num_auxiliary+1
    latency0{i}=num_auxiliary;
end
Opop1=pop0;
Ofit1=fit0;
Opop2=pop0;
Opop3=pop0;
Ofit2=fit0;
Tpop=pop0;
Tfit=fit0;
FE=N;
iter=1;
interval=2;
THETA2=5.*ones(2*num_auxiliary+1,Num_column-1);
THETA3=5.*ones(2*num_auxiliary+1,Num_column-1);
% rTHETA1=5.*ones(2*num_auxiliary+1,Num_column-1);
rTHETA2=5.*ones(2*num_auxiliary+1,Num_column-1);
rTHETA3=5.*ones(2*num_auxiliary+1,Num_column-1);
rdmodel2=cell(num_auxiliary+1,1);
Sample=Stratified_sample(Num_row,X0,num_auxiliary);
a=-0.5*cos(FE*pi/4000)+0.5;
b=1-a;
rnewpop1=cell(num_auxiliary+1,1); rnewpop2=cell(num_auxiliary+1,1);
rnewfit1=cell(num_auxiliary+1,1);  rnewfit2=cell(num_auxiliary+1,1); 
rapop=cell(num_auxiliary+1,1); rafit=cell(num_auxiliary+1,1);
rapopmeand=cell(num_auxiliary+1,1); raMSEd=cell(num_auxiliary+1,1); rapopmeanp=cell(num_auxiliary+1,1); raMSEp=cell(num_auxiliary+1,1); 
rdmodel2=cell(num_auxiliary+1,1);  wTfit=cell(num_auxiliary+1,1);  
uu=cell(num_auxiliary+1,1); ll=cell(num_auxiliary+1,1); atpop=cell(num_auxiliary+1,1); atfit=cell(num_auxiliary+1,1);
                                   
while FE<maxFE
      parfor i=1:num_auxiliary+1
          midTHETA2=5.*ones(2*num_auxiliary+1,Num_column-1); midTHETA3=5.*ones(2*num_auxiliary+1,Num_column-1);
          Popdec1=cell(num_auxiliary+1,1);Popobj1=cell(num_auxiliary+1,1);
          Popdec2=cell(num_auxiliary+1,1); Popobj2=cell(num_auxiliary+1,1); 
          Popdec3=cell(num_auxiliary+1,1); Dvalue=cell(num_auxiliary+1,1);
          dmodel2=cell(1,1); dmodel3=cell(num_auxiliary,1);
          XX2=cell(num_auxiliary+1,1); YY2=cell(num_auxiliary+1,1);
          MSE2=cell(num_auxiliary+1,1); MMSE2=cell(num_auxiliary+1,1); Mpopobj2=cell(num_auxiliary+1,1);
          length=[];
          fit2=cell(num_auxiliary+1,1); piror_u2=cell(num_auxiliary+1,1);
          newpop1=cell(num_auxiliary+1,1); newpop2=cell(num_auxiliary+1,1);
          xu2=cell(num_auxiliary+1,1); k2=cell(num_auxiliary+1,1);
          newfit1=cell(num_auxiliary+1,1);  newfit2=cell(num_auxiliary+1,1);newXX2=cell(num_auxiliary+1,1);
          apop=cell(num_auxiliary+1,1); afit=cell(num_auxiliary+1,1); apopmeand=cell(num_auxiliary+1,1); aMSEd=cell(num_auxiliary+1,1);
          if i~=num_auxiliary+1
              [~,distinct1]=unique(Opop1{i},'rows');
              Popdec1{i}=Opop1{i}(distinct1,:);
              Popobj1{i}=Ofit1{i}(distinct1,:);
              datalong1=size(Popdec1{i},1);
              if datalong1>L1{i}
                  [~,paixu1]=sort(Popobj1{i});
                  data1=paixu1(1:floor(L1{i}/2),:);
                  paixu1=paixu1(floor(L1{i}/2)+1:end,:);
                  index1=randperm(size(paixu1,1));
                  data1=[data1;paixu1(index1(1:L1{i}-floor(L1{i}/2)),:)];
                  Popdec1{i}=Popdec1{i}(data1,:);
                  Popobj1{i}=Popobj1{i}(data1,:);
              end
                [newpop1{i},newfit1{i}]=callGA0(Sample{i},Y,Popdec1{i},CR,MR,Popobj1{i},L1{i});
                [apop{i},afit{i}]=callGA(Sample{i},Y,newpop1{i},CR,MR,newfit1{i},latency0{i});
                rnewpop1{i}=newpop1{i};
                rnewfit1{i}=newfit1{i};
                rapop{i}=apop{i};
                rafit{i}=afit{i};
                rafit{i}=rafit{i}(:);

%               dmodel1{i}=dacefit(Popdec1{i},Popobj1{i},'regpoly1','corrgauss',THETA1(i,:),1e-5.*ones(1,Num_column-1),100.*ones(1,Num_column-1));
%               midTHETA1(i,:)=dmodel1{i}.theta;
%               rTHETA1(i,:)=midTHETA1(i,:);
%               w1=0;
%               YY1{i}=Popobj1{i};
%               while w1<5
%                  if w1==0
%                      newXX1{i}=pop_generate0(Popdec1{i},CR,MR,YY1{i});
%                  else
%                      newXX1{i}=pop_generate(XX1{i},CR,MR,YY1{i});
%                  end
%                  XX1{i}=[XX1{i};newXX1{i}];
%                   for j=1:size(XX1{i})
%                       [YY1{i}(j,:),~,MSE1{i}(j,:)]=predictor(XX1{i}(j,:),dmodel1{i});
%                   end
%                   w1=w1+1;
%               end
%                ra=a;
%                rb=b;
%                [MMSE1{i},~]=max(MSE1{i},[],1);
%                [Mpopobj1{i},~]=max(YY1{i},[],1);
%                fit1{i}=YY1{i}./repmat(Mpopobj1{i},size(YY1{i},1),1)*rb+MSE1{i}./repmat(MMSE1{i},size(MSE1{i},1),1)*ra;
%                piror_u1{i}=sort(fit1{i});
%                piror_u1{i}=piror_u1{i}(1:3,1);
%                for j=1:size(fit1{i},1)
%                    for k=1:3
%                        if fit1{i}(j,1)==piror_u1{i}(k,1)
%                            xu1{i}(j,1)=1;
%                            break;
%                        else
%                            xu1{i}(j,1)=0;
%                        end
%                    end
%                end
%                    k1{i}=sum(xu1{i}(:,1)==1);
%                for j=1:k1{i}
%                    for k=1:size(xu1{i},1)
%                        if xu1{i}(k,1)==1
%                            newpop1{i}(j,:)=XX1{i}(k,:);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
%                            xu1{i}(k,1)=0;
%                            break;
%                        end
%                    end
%                end
%               [newpop1{i},~]=unique(newpop1{i}(:,:),'rows');
%                if size(newpop1{i},1)<=3
%                    newpop1{i}=newpop1{i};
%                elseif size(newpop1{i},1)>3
%                    newpop1{i}=newpop1{i}(randperm(size(newpop1{i},1),3),:);
%                end
%                rnewpop1{i}=newpop1{i};
%                for j=1:size(newpop1{i},1)
%                    [newfit1{i}(j,1),~]=FitnessFunction(Sample{i},Y,newpop1{i}(j,:));
%                end
%                rnewfit1{i}=newfit1{i}; 
%                aN{i}=3*latency0{i};
%                [apop{i},afit{i}]=callGA(X0,Y,newpop1{i},CR,MR,newfit1{i},latency0{i});
%                afit{i}=afit{i}(:);
%                rapop{i}=apop{i};  rafit{i}=afit{i};
               
              for j=1:num_auxiliary
                  length(j,1)=size(Ofit2{j},1); 
              end
              K=size(Ofit2{num_auxiliary+1},1);
              length(num_auxiliary+1,1)=K;
              minl=min(length);
              for j=1:minl
                  Dvalue{i}(j,:)=Ofit2{num_auxiliary+1}(j,:)-Ofit2{i}(j,:);
              end
              Opop3{i}=Opop2{i}(1:minl,:);
              [~,distinct3]=unique(Opop3{i},"rows");
              Popdec3{i}=Opop3{i}(distinct3,:);
              Dvalue{i}=Dvalue{i}(distinct3);
              datalong3=size(Popdec3{i},1);
              if datalong3>L2{i}
                    [~,paixu3] = sort(Dvalue{i});
                    data3=paixu3(1:floor(L2{i}/2), :);
                    paixu3=paixu3(floor(L2{i}/2)+1:end,:);
                    index3=randperm(size(paixu3,1));
                    data3=[data3;paixu3(index3(1:L2{i}-floor(L2{i}/2)), :)];
                    Popdec3{i}=Popdec3{i}(data3,:);Dvalue{i}=Dvalue{i}(data3,:);
              end
   
              dmodel3{i}=dacefit(Popdec3{i},Dvalue{i},'regpoly1','corrgauss',THETA3(i,:),1e-5.*ones(1,Num_column-1),100.*ones(1,Num_column-1));
              midTHETA3(i,:)=dmodel3{i}.theta;
              rTHETA3(i,:)=midTHETA3(i,:);
              
              for j=1:size(apop{i},1)
                  [apopmeand{i}(j,:),~,aMSEd{i}(j,:)]=predictor(apop{i}(j,:),dmodel3{i});
              end
              rapopmeand{i}=apopmeand{i}; raMSEd{i}=aMSEd{i};
             
          end
           if i==num_auxiliary+1
              if(mod((iter-1),interval)==0)
                  [~,distinct2]=unique(Tpop{i},'rows');
                  Popdec2{i}=Tpop{i}(distinct2,:);
                  Popobj2{i}=Tfit{i}(distinct2,:);
                  datalong2=size(Popdec2{i},1);
              else
                  [~,distinct2]=unique(Opop2{i},'rows');
                  Popdec2{i}=Opop2{i}(distinct2,:);
                  Popobj2{i}=Ofit2{i}(distinct2,:);
                  datalong2=size(Popdec2{i},1);
              end
              if datalong2>L2{i}
                  [~,paixu2] = sort(Popobj2{i});
                  data2=paixu2(1:floor(L2{i}/2), :);
                  paixu2=paixu2(floor(L2{i}/2)+1:end,:);
                  index2=randperm(size(paixu2,1));
                  data2=[data2;paixu2(index2(1:L2{i}-floor(L2{i}/2)), :)];
                  Popdec2{i}=Popdec2{i}(data2,:);Popobj2{i}=Popobj2{i}(data2,:);
              end
              dmodel2=dacefit(Popdec2{i},Popobj2{i},'regpoly1','corrgauss',THETA2(i,:),1e-5.*ones(1,Num_column-1),100.*ones(1,Num_column-1));
              rdmodel2{i}=dmodel2;
              midTHETA2(i,:)=dmodel2.theta;
              rTHETA2(i,:)=midTHETA2(i,:);
              w2=0;
              YY2{i}=Popobj2{i};
              while w2<5
                  if w2==0
                      newXX2{i}=pop_generate0(Popdec2{i},CR,MR,YY2{i});
                  else
                      newXX2{i}=pop_generate(XX2{i},CR,MR,YY2{i});
                  end
                  
                  XX2{i}=[XX2{i};newXX2{i}];
                  for j=1:size(XX2{i})
                      [YY2{i}(j,:),~,MSE2{i}(j,:)]=predictor(XX2{i}(j,:),dmodel2);
                  end
                  w2=w2+1;
              end
               [MMSE2{i},~]=max(MSE2{i},[],1);
               [Mpopobj2{i},~]=max(YY2{i},[],1);
               ra=a;
               rb=b;
               fit2{i}=YY2{i}./repmat(Mpopobj2{i},size(YY2{i},1),1)*rb+MSE2{i}./repmat(MMSE2{i},size(MSE2{i},1),1)*ra;
               piror_u2{i}=sort(fit2{i});
               piror_u2{i}=piror_u2{i}(1:3,1);
            
                for j=1:size(fit2{i},1)
                   for k=1:size(piror_u2{i},1)
                       if fit2{i}(j,1)==piror_u2{i}(k,1)
                           xu2{i}(j,1)=1;
                           break;
                       else
                           xu2{i}(j,1)=0;
                       end
                   end
                end
               k2{i}=sum(xu2{i}(:,1)==1);
               for j=1:k2{i}
                   for k=1:size(XX2{i},1)
                       if xu2{i}(k,1)==1
                           newpop2{i}(j,:)=XX2{i}(k,:);
                           xu2{i}(k,1)=0;
                           break;
                       end
                   end
               end
               [newpop2{i},~]=unique(newpop2{i}(:,:),'rows');
               if size(newpop2{i},1)<=u
                   newpop2{i}=newpop2{i};
               elseif size(newpop2{i},1)>u
                   newpop2{i}=newpop2{i}(randperm(size(newpop2{i},1),u),:);
               end
               rnewpop2{i}=newpop2{i};
               for j=1:size(newpop2{i},1)
                   [newfit2{i}(j,1),~]=FitnessFunction(Sample{i},Y,newpop2{i}(j,:));
               end
               rnewfit2{i}=new
               
               fit2{i};
           end
      end
          THETA2=rTHETA2;
          THETA3=rTHETA3;
          for j=1:num_auxiliary+1
              FE=FE+size(rnewpop1{j},1)+size(rnewpop2{j},1);
          end
          FE=FE-u;
          for j=1:num_auxiliary
              for k=1:size(rapop{j},1)
                  [rapopmeanp{j}(k,:),~,raMSEp{j}(k,:)]=predictor(rapop{j}(k,:),rdmodel2{num_auxiliary+1});
              end
          end
          for j=1:num_auxiliary
              for k=1:size(rapop{j},1)
                  wTfit{j}(k,:)=rafit{j}(k,:)+rapopmeand{j}(k,:);
              end
          end
              for j=1:num_auxiliary
                  length1(j,1)=size(rapop{j},1); 
              end
          for j=1:num_auxiliary
              for k=1:length1(j,1)
                  uu{j}(k,:)=rapopmeanp{j}(k,:)+raMSEp{j}(k,:);
                  ll{j}(k,:)=rapopmeanp{j}(k,:)-raMSEp{j}(k,:);
                  ff=find(ll{j}(k,:)<=wTfit{j}(k,:)&wTfit{j}(k,:)<=uu{j}(k,:));
                  atpop{j}=rapop{j}(ff,:);
                  atfit{j}=wTfit{j}(ff,:);
              end
          end
          for j=1:num_auxiliary
              length2(j,1)=size(rnewpop1{j},1);
          end
          min2=min(length2);
          for j=1:num_auxiliary+1
                  Opop1{j}=[Opop1{j};rnewpop1{j};rapop{j}];
                  Ofit1{j}=[Ofit1{j};rnewfit1{j};rafit{j}];
                   Tpop{j}=[Tpop{j};rnewpop2{j};atpop{j}];
                   Tfit{j}=[Tfit{j};rnewfit2{j};atfit{j}];
          end
          for j=1:num_auxiliary
              for k=1:min2
                  Opop2{j}=[Opop2{j};rnewpop1{j}(k,:)];
                  Ofit2{j}=[Ofit2{j};rnewfit1{j}(k,:)];
              end
          end
          Opop2{num_auxiliary+1}=[Opop2{num_auxiliary+1};rnewpop2{num_auxiliary+1}];
          Ofit2{num_auxiliary+1}=[Ofit2{num_auxiliary+1};rnewfit2{num_auxiliary+1}];
          a=-0.5*cos(FE*pi/4000)+0.5;
          b=1-a;
          N=N+size(rnewpop2{num_auxiliary+1},1);
          if(mod((iter-1),interval)==0)
            for j=1:num_auxiliary
                length3(j,1)=size(atpop{j},1);
            end
            tnum=sum(length3,1);
%             if tnum==0 
%                 for j=1:num_auxiliary
%                     L1{j}=randi([15,25]);
%                 end
%                 Lsum=0;
%                 for j=1:num_auxiliary
%                     Lsum=Lsum+size(L1{j});
%                 end
%                 L2{num_auxiliary+1}=N-Lsum;
%             else
                for j=1:num_auxiliary
                    L1{j}=ceil(N*length3(j,1)/tnum*length3(j,1)/length1(j,1));
                end
                L1sum=0;
                for j=1:num_auxiliary
                    L1sum=L1sum+size(L1{j});
                end
                L2{num_auxiliary+1}=N-L1sum;
%             end
          end
gbest=[];
for j=1:num_auxiliary
    Ofit1{j}(all(Ofit1{j}==0,2),:)=[];
end
for j=1:num_auxiliary+1
    if j<num_auxiliary+1
       gbest(j,1)=min(Ofit1{j});
    else
       gbest(j,1)=min(Tfit{j});
    end
end
gbest=min(gbest);
Gbest(iter)=1-gbest;
FE1(iter)  =FE; 
%  for j=1:3
%      num00{j}(iter)=L1{j};
%  end
%  num00{4}(iter)=L2{4}(1,1);
iter=iter+1;
end      
PF_Time(1,1) = etime(clock,t);
% average_fitG=sum(Gbest,1)/3;
% average_time=sum(PF_Time,1)/3;
end
% Gbest=sum(Gbest,1)/3;
% FE1=sum(FE1,1)/3;
% PF_Time=sum(PF_Time)/3;



      