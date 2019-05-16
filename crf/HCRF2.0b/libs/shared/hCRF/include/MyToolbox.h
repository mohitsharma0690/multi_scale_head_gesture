//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Trainer Component
//
//	June 18, 2007

#ifndef __MY_TOOLBOX_H
#define __MY_TOOLBOX_H

#include "toolbox.h"
#include "MyFeatures.h"

class MyToolbox: public ToolboxCRF
{
public:
	MyToolbox();
	MyToolbox(int opt, int windowSize = 0);
	~MyToolbox();

private:
	void init(int opt);
};

#endif
