load L22_14Vx.mat;
elePos = Trans.ElementPos;
f0 = Trans.frequency*1e6;   
Fs = 4*f0; 
c = 1540;
chano = Trans.numelements;
m2wl = Trans.frequency*1e6/c;
angles = deg2rad(-2:1:2);
NoAngles   = length(angles);

%% Define element Pos
x = elePos(1,1):1:elePos(end,1);
arrayx = elePos(:,1)/m2wl;
z = 0:1/2:(12e-3*m2wl);
x = x/m2wl;
z = z/m2wl;
[X,Z]      = meshgrid(x,z);
[AxialDim,LateralDim]    = size(X);
ElevationDim = 1;

%% Transmit Delay
twpeak = 2.4992/m2wl/c;
txDelay = single(zeros(AxialDim,LateralDim,ElevationDim,NoAngles));

for n = 1:NoAngles
    Tdelay = X * sin(angles(n))/c;
    Tdelay = Tdelay - min(Tdelay(:)) + Z * cos(angles(n))/c;
    txDelay(:,:,:,n) = (Tdelay + twpeak)*Fs; 
end

delay = reshape(txDelay,[AxialDim*LateralDim*ElevationDim*NoAngles,1]);
f1 = fopen('\L22_14Vx\txDelay.dat', 'w');
fwrite(f1, delay, 'single');
fclose(f1);

%% Receive Delay
arrayz     = zeros(1,chano);
X          = repmat(X,[1,1,1,chano]);
Z          = repmat(Z,[1,1,1,chano]);
arrayX     = repmat(reshape(arrayx,[1,1,1,chano]),[AxialDim,LateralDim,1]);
arrayZ     = repmat(reshape(arrayz,[1,1,1,chano]),[AxialDim,LateralDim,1]);


lensCorrection = c /m2wl * (0.05/2145 + 0.48/1147);
rcvDelay = (sqrt((Z-arrayZ).^2+(X-arrayX).^2)+lensCorrection)/c;
rcvDelay = single(rcvDelay*Fs);     

delay = reshape(rcvDelay,[AxialDim*LateralDim*ElevationDim*chano,1]);
f1 = fopen('\L22_14Vx\rcvDelay.dat', 'w');
fwrite(f1, delay, 'single');
fclose(f1);


%% Element Sensitivity
senscutoff = 0.6;
Theta = abs(atan((X-arrayX)./(Z-arrayZ))); 
XT    = Trans.elementWidth*pi*sin(Theta);
ElementSens = abs(cos(Theta).*(sin(XT)./XT));
ElementSens(ElementSens<senscutoff)  = 0;
ElementSens(ElementSens>=senscutoff) = 1;
ElementSens(isnan(ElementSens))      = 1;
ElementSens = single(ElementSens);


elementSens = reshape(ElementSens,[AxialDim*LateralDim*ElevationDim*chano,1]);
f1 = fopen('\L22_14Vx\elementSens.dat', 'w');
fwrite(f1, elementSens, 'single');
fclose(f1);