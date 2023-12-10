clc
clear all
close all
format bank;
x=input('Enter the x coordinate sequence:');
y=input('Enter the y coordinate sequence:');
k=input('Enter the number of clusters:');
cx=input('Enter the  initial means x coordinate:');%initial mean of x
cy=input('Enter the  initial means y coordinate:');%initial mean of y
mean_oldx=cx;
mean_newx=cx;
mean_oldy=cy;
mean_newy=cy;
outputx=cell(k,1);
outputy=cell(k,1);
temp=0;
while(temp==0)
    mean_oldx=mean_newx;
    mean_oldy=mean_newy;
    for ij=1:length(x)
        mina=[];
        mu=x(ij);
        nu=y(ij);
     for mk=1:length(cx)
         mina=[mina sqrt((mu-cx(mk))^2+(nu-cy(mk))^2)];
     end
     [gc index]=min(mina);
     outputx{index}=[outputx{index} mu];
     outputy{index}=[outputy{index} nu];
    end
    gmckx=[];
    gmcky=[];
    for i=1:k
        gmckx=[gmckx mean(outputx{i})];
        gmcky=[gmcky mean(outputy{i})];
    end
    cx=gmckx;
    cy=gmcky;
    mean_newx=cx;
        mean_newy=cy;
        gum=0;
        bum=0;
    if(mean_newx==mean_oldx)
        gum=1;
    end
    if(mean_newy==mean_oldy)
        bum=1;
    end
    if(gum==1 && bum==1)
        temp=1;
    else
            outputx=cell(k,1);
            outputy=cell(k,1);
    end
    
end
celldisp(outputx);
celldisp(outputy);
gm=rand(1,k);
tm=rand(1,k);
bm=rand(1,k);
bg=1;
for i=1:k
    x=outputx{i};
    y=outputy{i};
    for j=1:length(x)
        t=0:0.01:2*pi;
        a=x(j)+1000*cos(t);
        b=y(j)+1000*sin(t);
        patch(a,b,[gm(bg),tm(bg),bm(bg)]);
        axis square;
        hold on;
    end
    bg=bg+1;
    hold on;
    axis square;
    grid on;
end