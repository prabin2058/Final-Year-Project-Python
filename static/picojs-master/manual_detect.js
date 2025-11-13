
var manualDetect = {}

manualDetect.rgbaToGrayscale = function(rgba, nrows, ncols)
{
	var gray = new Uint8Array(nrows*ncols)
	for(var r=0; r<nrows; ++r)
		for(var c=0; c<ncols; ++c)
			gray[r*ncols + c] = (2*rgba[r*4*ncols+4*c+0]+7*rgba[r*4*ncols+4*c+1]+1*rgba[r*4*ncols+4*c+2])/10
	return gray
}

manualDetect.unpackCascade = function(bytes)
{
	var dview = new DataView(new ArrayBuffer(4))
	var p = 8
	dview.setUint8(0, bytes[p+0]); dview.setUint8(1, bytes[p+1]); dview.setUint8(2, bytes[p+2]); dview.setUint8(3, bytes[p+3])
	var tdepth = dview.getInt32(0, true)
	p = p + 4
	dview.setUint8(0, bytes[p+0]); dview.setUint8(1, bytes[p+1]); dview.setUint8(2, bytes[p+2]); dview.setUint8(3, bytes[p+3])
	var ntrees = dview.getInt32(0, true)
	p = p + 4
	var tcodes_ls = []
	var tpreds_ls = []
	var thresh_ls = []
	for(var t=0; t<ntrees; ++t)
	{
		Array.prototype.push.apply(tcodes_ls, [0, 0, 0, 0])
		Array.prototype.push.apply(tcodes_ls, bytes.slice(p, p+4*Math.pow(2, tdepth)-4))
		p = p + 4*Math.pow(2, tdepth)-4
		for(var i=0; i<Math.pow(2, tdepth); ++i)
		{
			dview.setUint8(0, bytes[p+0]); dview.setUint8(1, bytes[p+1]); dview.setUint8(2, bytes[p+2]); dview.setUint8(3, bytes[p+3])
			tpreds_ls.push(dview.getFloat32(0, true))
			p = p + 4
		}
		dview.setUint8(0, bytes[p+0]); dview.setUint8(1, bytes[p+1]); dview.setUint8(2, bytes[p+2]); dview.setUint8(3, bytes[p+3])
		thresh_ls.push(dview.getFloat32(0, true))
		p = p + 4
	}
	var tcodes = new Int8Array(tcodes_ls)
	var tpreds = new Float32Array(tpreds_ls)
	var thresh = new Float32Array(thresh_ls)
	function classifyRegion(r, c, s, pixels, ldim)
	{
		r = 256*r
		c = 256*c
		var root = 0
		var o = 0.0
		var pow2tdepth = Math.pow(2, tdepth) >> 0
		for(var i=0; i<ntrees; ++i)
		{
			var idx = 1
			for(var j=0; j<tdepth; ++j)
				idx = 2*idx + (pixels[((r+tcodes[root + 4*idx + 0]*s) >> 8)*ldim+((c+tcodes[root + 4*idx + 1]*s) >> 8)]<=pixels[((r+tcodes[root + 4*idx + 2]*s) >> 8)*ldim+((c+tcodes[root + 4*idx + 3]*s) >> 8)])
			o = o + tpreds[pow2tdepth*i + idx-pow2tdepth]
			if(o<=thresh[i]) return -1
			root += 4*pow2tdepth
		}
		return o - thresh[ntrees-1]
	}
	return classifyRegion
}

manualDetect.runCascade = function(image, classifyRegion, params)
{
	var pixels = image.pixels
	var nrows = image.nrows
	var ncols = image.ncols
	var ldim = image.ldim
	var shiftfactor = params.shiftfactor
	var minsize = params.minsize
	var maxsize = params.maxsize
	var scalefactor = params.scalefactor
	var scale = minsize
	var detections = []
	while(scale<=maxsize)
	{
		var step = Math.max(shiftfactor*scale, 1) >> 0
		var offset = (scale/2 + 1) >> 0
		for(var r=offset; r<=nrows-offset; r+=step)
			for(var c=offset; c<=ncols-offset; c+=step)
			{
				var q = classifyRegion(r, c, scale, pixels, ldim)
				if(q>0.0) detections.push([r, c, scale, q])
			}
		scale = scale*scalefactor
	}
	return detections
}

manualDetect.clusterDetections = function(dets, iouthreshold)
{
	dets = dets.sort(function(a,b){ return b[3]-a[3] })
	function iou(det1, det2)
	{
		var r1=det1[0], c1=det1[1], s1=det1[2]
		var r2=det2[0], c2=det2[1], s2=det2[2]
		var overr = Math.max(0, Math.min(r1+s1/2, r2+s2/2) - Math.max(r1-s1/2, r2-s2/2))
		var overc = Math.max(0, Math.min(c1+s1/2, c2+s2/2) - Math.max(c1-s1/2, c2-s2/2))
		return overr*overc/(s1*s1+s2*s2-overr*overc)
	}
	var assignments = new Array(dets.length).fill(0)
	var clusters = []
	for(var i=0;i<dets.length;++i)
	{
		if(assignments[i]==0)
		{
			var r=0.0, c=0.0, s=0.0, q=0.0, n=0
			for(var j=i;j<dets.length;++j)
				if(iou(dets[i], dets[j])>iouthreshold)
				{
					assignments[j]=1
					r+=dets[j][0]; c+=dets[j][1]; s+=dets[j][2]; q+=dets[j][3]; n+=1
				}
			clusters.push([r/n, c/n, s/n, q])
		}
	}
	return clusters
}

manualDetect.instantiateMemory = function(size)
{
	var n = 0
	var memory = []
	for(var i=0;i<size;++i) memory.push([])
	return function(dets)
	{
		memory[n] = dets
		n = (n+1)%memory.length
		dets = []
		for(var k=0;k<memory.length;++k) dets = dets.concat(memory[k])
		return dets
	}
}

if (typeof module !== 'undefined') module.exports = manualDetect


