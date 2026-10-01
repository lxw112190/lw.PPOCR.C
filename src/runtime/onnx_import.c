/* Pure-C bounded protobuf reader, inspired by lw.PPOCR.Vulkan (Apache-2.0).
 * Targets our existing NCHW semantic IR, preserving LWM compatibility. */
#include "session_internal.h"
#include "lwm_read.h"
#include <limits.h>
#include <math.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define OX_MAX_VALUES 16384u
#define OX_MAX_NODES 4096u
#define OX_MAX_ATTRS 16u
#define OX_MAX_BYTES LW_ONNX_MAX_BYTES
#define OX_NONE UINT32_MAX
typedef struct ox_reader { const uint8_t* p; const uint8_t* end; } ox_reader;
typedef struct ox_attr {
    char name[64], string[128];
    int64_t ints[8], i;
    uint32_t count, type;
    float f;
} ox_attr;
typedef struct ox_value {
    char* name;
    uint32_t alias, defined, constant, ready, metadata, used, index, first_rank;
    lw_runtime_tensor shape;
    int32_t first[8], dynamic[8];
    const uint8_t* raw;
    uint8_t* owned;
    size_t bytes;
    int64_t controls[8];
    uint32_t control_count;
} ox_value;
typedef struct ox_node {
    char op[48];
    uint32_t inputs[8], output, count, attr_count, emitted;
    ox_attr attrs[OX_MAX_ATTRS];
    uint8_t params[136];
    uint32_t param_size, lowered_count;
    uint16_t opcode;
} ox_node;
typedef struct ox_graph {
    ox_value* values;
    ox_node* nodes;
    uint32_t value_count, node_count, value_capacity, node_capacity, input, output, opset;
    size_t names_bytes;
    uint64_t constant_bytes;
    lw_runtime_tensor declared_output;
    jmp_buf* escape;
    lw_error* error;
    lw_status status;
} ox_graph;
/* All parse failures unwind to one cleanup point. Buffers tracked in ox_graph
 * remain owned there until normalization succeeds; no partial model escapes. */
#if defined(_MSC_VER)
__declspec(noreturn)
#else
_Noreturn
#endif
static void ox_bad(ox_graph* g,lw_status status,const char* message) {
    g->status=status; lw_set_error(g->error,status,message); longjmp(*g->escape,1);
}
#define OX_REQUIRE(g,c,m) do { if(!(c)) ox_bad((g),LW_STATUS_INVALID_FORMAT,(m)); } while(0)
static uint64_t ox_var(ox_graph* g,ox_reader* r) {
    uint64_t v=0u; unsigned i;
    for(i=0u;i<10u;++i) {
        uint8_t b;
        OX_REQUIRE(g,r->p!=r->end,"ONNX truncated varint"); b=*r->p++;
        OX_REQUIRE(g,i!=9u || b<=1u,"ONNX varint overflow");
        v|=(uint64_t)(b&127u)<<(7u*i);
        if((b&128u)==0u) return v;
    }
    ox_bad(g,LW_STATUS_INVALID_FORMAT,"ONNX invalid varint");
}
static ox_reader ox_child(ox_graph* g,ox_reader* r) {
    uint64_t n=ox_var(g,r); ox_reader c;
    OX_REQUIRE(g,n<=(uint64_t)(r->end-r->p),"ONNX length exceeds message");
    c.p=r->p; c.end=r->p+(size_t)n; r->p=c.end; return c;
}
static uint32_t ox_fixed(ox_graph* g,ox_reader* r) {
    uint32_t v; OX_REQUIRE(g,r->end-r->p>=4,"ONNX truncated fixed32");
    v=lwm_read_u32(r->p); r->p+=4; return v;
}
static int ox_tag(ox_graph* g,ox_reader* r,unsigned* f,unsigned* w) {
    uint64_t t; if(r->p==r->end) return 0; t=ox_var(g,r);
    OX_REQUIRE(g,t<=UINT32_MAX && (t>>3u)!=0u,"ONNX invalid tag");
    *f=(unsigned)(t>>3u); *w=(unsigned)(t&7u); return 1;
}
static void ox_skip(ox_graph* g,ox_reader* r,unsigned w) {
    if(w==0u) { (void)ox_var(g,r); return; }
    if(w==2u) { (void)ox_child(g,r); return; }
    if(w==1u || w==5u) {
        size_t n=w==1u?8u:4u;
        OX_REQUIRE(g,(size_t)(r->end-r->p)>=n,"ONNX truncated fixed field"); r->p+=n; return;
    }
    ox_bad(g,LW_STATUS_INVALID_FORMAT,"ONNX unsupported wire type");
}
static void ox_string(ox_graph* g,ox_reader* r,char* dst,size_t cap) {
    ox_reader c=ox_child(g,r); size_t n=(size_t)(c.end-c.p);
    OX_REQUIRE(g,n<cap && memchr(c.p,0,n)==NULL,"ONNX invalid/oversized string");
    memcpy(dst,c.p,n); dst[n]=0;
}
static uint32_t ox_value_id(ox_graph* g,const char* name) {
    uint32_t i; size_t n=strlen(name)+1u;
    OX_REQUIRE(g,n>1u,"ONNX empty tensor name");
    for(i=0u;i<g->value_count;++i) if(strcmp(g->values[i].name,name)==0) return i;
    OX_REQUIRE(g,g->value_count<g->value_capacity && n<=8u*1024u*1024u-g->names_bytes,"ONNX name budget");
    i=g->value_count++; g->values[i].name=(char*)malloc(n);
    if(g->values[i].name==NULL) ox_bad(g,LW_STATUS_OUT_OF_MEMORY,"ONNX name allocation");
    memcpy(g->values[i].name,name,n); g->names_bytes+=n; g->values[i].alias=OX_NONE; return i;
}
static uint32_t ox_root(ox_graph* g,uint32_t id) {
    uint32_t steps=0u;
    while(g->values[id].alias!=OX_NONE) {
        OX_REQUIRE(g,++steps<=g->value_count,"ONNX Identity cycle"); id=g->values[id].alias;
    }
    return id;
}
static void ox_put16(uint8_t* p,uint16_t v) { p[0]=(uint8_t)v; p[1]=(uint8_t)(v>>8); }
static void ox_put32(uint8_t* p,uint32_t v) { unsigned i; for(i=0u;i<4u;++i) p[i]=(uint8_t)(v>>(8u*i)); }
static void ox_put64(uint8_t* p,uint64_t v) { unsigned i; for(i=0u;i<8u;++i) p[i]=(uint8_t)(v>>(8u*i)); }
static void ox_float(uint8_t* p,float v) { uint32_t b; memcpy(&b,&v,4u); ox_put32(p,b); }
static uint32_t ox_dtype(ox_graph* g,uint32_t t) {
    if(t==1u) return LW_DTYPE_F32;
    if(t==6u) return LW_DTYPE_I32;
    if(t==7u) return LW_DTYPE_I64;
    if(t==2u) return LW_DTYPE_U8;
    ox_bad(g,LW_STATUS_UNSUPPORTED,"ONNX unsupported dtype");
}
/* Initializers borrow raw bytes during import. Only typed payloads need a
 * temporary allocation; the normalized IR owns its final aligned copy. */
static void ox_tensor(ox_graph* g,ox_reader r) {
    char name[4097]={0}; int32_t dims[8]={0};
    uint32_t rank=0u,type=0u,id,i; const uint8_t* raw=NULL;
    ox_reader typed={NULL,NULL}; unsigned typed_field=0u,f,w;
    size_t raw_size=0u; uint64_t count=1u,bytes; ox_value* v;
    while(ox_tag(g,&r,&f,&w)) {
        if(f==1u && (w==0u || w==2u)) {
            ox_reader c=w==2u?ox_child(g,&r):r;
            while(c.p!=c.end) {
                uint64_t d=ox_var(g,&c);
                OX_REQUIRE(g,rank<8u && d<=INT32_MAX,"ONNX initializer dimensions");
                dims[rank++]=(int32_t)d;
                if(w==0u) break;
            }
            if(w==0u) r=c;
        } else if(f==2u && w==0u) {
            uint64_t d=ox_var(g,&r);
            OX_REQUIRE(g,type==0u && d<=UINT32_MAX,"ONNX duplicate/invalid dtype"); type=(uint32_t)d;
        } else if(f==8u && w==2u) ox_string(g,&r,name,sizeof(name));
        else if(f==9u && w==2u) {
            ox_reader c=ox_child(g,&r);
            OX_REQUIRE(g,raw==NULL,"ONNX duplicate raw_data"); raw=c.p; raw_size=(size_t)(c.end-c.p);
        } else if((f==4u || f==5u || f==7u) && w==2u) {
            OX_REQUIRE(g,typed.p==NULL,"ONNX repeated typed payload");
            typed=ox_child(g,&r); typed_field=f;
        } else if(f==14u && w==0u) {
            OX_REQUIRE(g,ox_var(g,&r)==0u,"ONNX external tensor data unsupported");
        } else if(f==13u || f==14u)
            ox_bad(g,LW_STATUS_UNSUPPORTED,"ONNX external tensor data unsupported");
        else ox_skip(g,&r,w);
    }
    id=ox_value_id(g,name); v=&g->values[id];
    OX_REQUIRE(g,!v->defined,"ONNX duplicate initializer/producer");
    v->defined=1u; v->constant=1u; v->shape.dtype=ox_dtype(g,type); v->shape.rank=rank;
    memcpy(v->shape.dimensions,dims,sizeof(dims));
    for(i=0u;i<rank;++i) {
        OX_REQUIRE(g,dims[i]==0 || count<=OX_MAX_BYTES/(uint32_t)dims[i],"ONNX tensor overflow");
        count*=(uint32_t)dims[i];
    }
    bytes=count*(type==7u?8u:type==2u?1u:4u);
    OX_REQUIRE(g,bytes<=OX_MAX_BYTES && bytes<=OX_MAX_BYTES-g->constant_bytes,"ONNX constant budget");
    g->constant_bytes+=bytes; v->bytes=(size_t)bytes;
    if(raw!=NULL) {
        OX_REQUIRE(g,typed.p==NULL && raw_size==v->bytes,"ONNX raw tensor size mismatch"); v->raw=raw;
    } else if(bytes!=0u) {
        size_t j;
        OX_REQUIRE(g,typed.p!=NULL,"ONNX missing tensor data");
        v->owned=(uint8_t*)malloc(v->bytes);
        if(v->owned==NULL) ox_bad(g,LW_STATUS_OUT_OF_MEMORY,"ONNX tensor allocation");
        v->raw=v->owned;
        for(j=0u;j<(size_t)count;++j) {
            if(type==1u) {
                OX_REQUIRE(g,typed_field==4u,"ONNX float payload mismatch");
                ox_put32(v->owned+j*4u,ox_fixed(g,&typed));
            } else {
                uint64_t x;
                OX_REQUIRE(g,typed_field==(type==7u?7u:5u),"ONNX integer payload mismatch");
                x=ox_var(g,&typed);
                if(type==7u) ox_put64(v->owned+j*8u,x);
                else if(type==2u) {
                    OX_REQUIRE(g,x<=UINT8_MAX,"ONNX uint8 payload overflow");
                    v->owned[j]=(uint8_t)x;
                } else {
                    OX_REQUIRE(g,(int64_t)x>=INT32_MIN && (int64_t)x<=INT32_MAX,
                               "ONNX int32 payload overflow");
                    ox_put32(v->owned+j*4u,(uint32_t)x);
                }
            }
        }
        OX_REQUIRE(g,typed.p==typed.end,"ONNX surplus typed data");
    }
    if(type==1u) {
        size_t j;
        for(j=0u;j<v->bytes;j+=4u) {
            uint32_t b=lwm_read_u32(v->raw+j); float x; memcpy(&x,&b,4u);
            OX_REQUIRE(g,isfinite(x),"ONNX nonfinite constant");
        }
    }
    if((type==6u || type==7u) && count<=8u) {
        v->metadata=1u; v->control_count=(uint32_t)count;
        for(i=0u;i<(uint32_t)count;++i)
            v->controls[i]=type==7u?(int64_t)lwm_read_u64(v->raw+i*8u):(int32_t)lwm_read_u32(v->raw+i*4u);
    }
}
static void ox_attribute(ox_graph* g,ox_reader r,ox_attr* a) {
    unsigned f,w;
    while(ox_tag(g,&r,&f,&w)) {
        if(f==1u && w==2u) ox_string(g,&r,a->name,sizeof(a->name));
        else if(f==20u && w==0u) {
            uint64_t t=ox_var(g,&r); OX_REQUIRE(g,t<=UINT32_MAX,"ONNX attribute type overflow"); a->type=(uint32_t)t;
        } else if(f==2u && w==5u) {
            uint32_t b=ox_fixed(g,&r); memcpy(&a->f,&b,4u);
            OX_REQUIRE(g,isfinite(a->f),"ONNX nonfinite attribute");
        } else if(f==3u && w==0u) a->i=(int64_t)ox_var(g,&r);
        else if(f==4u && w==2u) ox_string(g,&r,a->string,sizeof(a->string));
        else if(f==8u && (w==0u || w==2u)) {
            ox_reader c=w==2u?ox_child(g,&r):r;
            while(c.p!=c.end) {
                OX_REQUIRE(g,a->count<8u,"ONNX attribute list budget");
                a->ints[a->count++]=(int64_t)ox_var(g,&c);
                if(w==0u) break;
            }
            if(w==0u) r=c;
        } else if(f==5u || f==6u || f==10u || f==11u)
            ox_bad(g,LW_STATUS_UNSUPPORTED,"ONNX tensor/graph attribute unsupported");
        else ox_skip(g,&r,w);
    }
    OX_REQUIRE(g,a->name[0]!=0 && (a->type==1u || a->type==2u || a->type==3u || a->type==7u),
               "ONNX unsupported attribute type");
}
static void ox_parse_node(ox_graph* g,ox_reader r) {
    unsigned f,w; char name[4097]; ox_node* n; uint32_t i;
    OX_REQUIRE(g,g->node_count<g->node_capacity,"ONNX node budget");
    n=&g->nodes[g->node_count++]; n->output=OX_NONE;
    while(ox_tag(g,&r,&f,&w)) {
        if(f==1u && w==2u) {
            OX_REQUIRE(g,n->count<8u,"ONNX input arity budget");
            ox_string(g,&r,name,sizeof(name)); n->inputs[n->count++]=name[0]==0?OX_NONE:ox_value_id(g,name);
        } else if(f==2u && w==2u) {
            OX_REQUIRE(g,n->output==OX_NONE,"ONNX multiple node outputs unsupported");
            ox_string(g,&r,name,sizeof(name)); n->output=ox_value_id(g,name);
        } else if(f==4u && w==2u) ox_string(g,&r,n->op,sizeof(n->op));
        else if(f==5u && w==2u) {
            OX_REQUIRE(g,n->attr_count<OX_MAX_ATTRS,"ONNX attribute budget");
            ox_attribute(g,ox_child(g,&r),&n->attrs[n->attr_count]);
            for(i=0u;i<n->attr_count;++i)
                OX_REQUIRE(g,strcmp(n->attrs[i].name,n->attrs[n->attr_count].name)!=0,"ONNX duplicate attribute");
            ++n->attr_count;
        } else if(f==7u && w==2u) {
            ox_string(g,&r,name,sizeof(name));
            OX_REQUIRE(g,name[0]==0 || strcmp(name,"ai.onnx")==0,"ONNX custom domain");
        } else ox_skip(g,&r,w);
    }
    OX_REQUIRE(g,n->op[0]!=0 && n->output!=OX_NONE && n->count!=0u,"ONNX incomplete node");
    OX_REQUIRE(g,!g->values[n->output].defined,"ONNX duplicate producer"); g->values[n->output].defined=1u;
}
static void ox_shape_proto(ox_graph* g,ox_reader r,lw_runtime_tensor* t) {
    unsigned f,w;
    while(ox_tag(g,&r,&f,&w)) {
        if(f==1u && w==2u) {
            ox_reader d=ox_child(g,&r); unsigned df,dw; int32_t v=-1;
            OX_REQUIRE(g,t->rank<8u,"ONNX input rank budget");
            while(ox_tag(g,&d,&df,&dw)) {
                if(df==1u && dw==0u) {
                    uint64_t x=ox_var(g,&d);
                    OX_REQUIRE(g,x<=INT32_MAX,"ONNX input dimension overflow"); v=x==0u?-1:(int32_t)x;
                } else ox_skip(g,&d,dw);
            }
            t->dimensions[t->rank++]=v;
        } else ox_skip(g,&r,w);
    }
}
static uint32_t ox_value_info(ox_graph* g,ox_reader r,int input) {
    unsigned f,w; char name[4097]={0}; lw_runtime_tensor t; uint32_t id;
    memset(&t,0,sizeof(t));
    while(ox_tag(g,&r,&f,&w)) {
        if(f==1u && w==2u) ox_string(g,&r,name,sizeof(name));
        else if(f==2u && w==2u) {
            ox_reader type=ox_child(g,&r); unsigned tf,tw;
            while(ox_tag(g,&type,&tf,&tw)) {
                if(tf==1u && tw==2u) {
                    ox_reader tensor=ox_child(g,&type); unsigned vf,vw;
                    while(ox_tag(g,&tensor,&vf,&vw)) {
                        if(vf==1u && vw==0u) {
                            uint64_t dtype=ox_var(g,&tensor);
                            OX_REQUIRE(g,dtype<=UINT32_MAX,"ONNX ValueInfo dtype overflow");
                            t.dtype=ox_dtype(g,(uint32_t)dtype);
                        }
                        else if(vf==2u && vw==2u) ox_shape_proto(g,ox_child(g,&tensor),&t);
                        else ox_skip(g,&tensor,vw);
                    }
                } else ox_skip(g,&type,tw);
            }
        } else ox_skip(g,&r,w);
    }
    id=ox_value_id(g,name);
    if(input) {
        OX_REQUIRE(g,t.dtype==LW_DTYPE_F32 && t.rank==4u && t.dimensions[1]==3 &&
                       (t.dimensions[0]==-1 || t.dimensions[0]==1),"ONNX requires batch-one NCHW RGB input");
        OX_REQUIRE(g,!g->values[id].defined,"ONNX initializer-as-input unsupported");
        g->values[id].defined=1u; g->values[id].shape=t;
    } else {
        OX_REQUIRE(g,t.dtype!=0u,"ONNX output type is required");
        g->declared_output=t;
    }
    return id;
}
static void ox_parse_graph(ox_graph* g,ox_reader r) {
    unsigned f,w;
    while(ox_tag(g,&r,&f,&w)) {
        if(f==1u && w==2u) ox_parse_node(g,ox_child(g,&r));
        else if(f==5u && w==2u) ox_tensor(g,ox_child(g,&r));
        else if(f==11u && w==2u) {
            OX_REQUIRE(g,g->input==OX_NONE,"ONNX multiple inputs unsupported"); g->input=ox_value_info(g,ox_child(g,&r),1);
        } else if(f==12u && w==2u) {
            OX_REQUIRE(g,g->output==OX_NONE,"ONNX multiple outputs unsupported"); g->output=ox_value_info(g,ox_child(g,&r),0);
        } else if(f==15u) ox_bad(g,LW_STATUS_UNSUPPORTED,"ONNX sparse initializers unsupported");
        else ox_skip(g,&r,w);
    }
}
static ox_attr* ox_attr_find(ox_node* n,const char* name) {
    uint32_t i; for(i=0u;i<n->attr_count;++i) if(strcmp(n->attrs[i].name,name)==0) return &n->attrs[i];
    return NULL;
}
static int32_t ox_int(ox_graph* g,ox_node* n,const char* name,int32_t fallback) {
    ox_attr* a=ox_attr_find(n,name);
    if(a==NULL) return fallback;
    OX_REQUIRE(g,a->type==2u && a->i>=INT32_MIN && a->i<=INT32_MAX,"ONNX integer attribute invalid");
    return (int32_t)a->i;
}
static void ox_list(ox_graph* g,ox_node* n,const char* name,uint8_t* p,uint32_t count,int32_t fallback) {
    ox_attr* a=ox_attr_find(n,name); uint32_t i;
    OX_REQUIRE(g,a==NULL || (a->type==7u && a->count==count),"ONNX list attribute length invalid");
    for(i=0u;i<count;++i) {
        int64_t x=a==NULL?fallback:a->ints[i];
        OX_REQUIRE(g,x>=INT32_MIN && x<=INT32_MAX,"ONNX list attribute overflow");
        ox_put32(p+4u*i,(uint32_t)(int32_t)x);
    }
}
static float ox_fattr(ox_graph* g,ox_node* n,const char* name,float fallback) {
    ox_attr* a=ox_attr_find(n,name);
    OX_REQUIRE(g,a==NULL || a->type==1u,"ONNX float attribute invalid");
    return a==NULL?fallback:a->f;
}
static void ox_axes(ox_graph* g,ox_node* n,const char* name,uint32_t offset) {
    ox_attr* a=ox_attr_find(n,name); uint32_t count=a==NULL?0u:a->count;
    ox_put16(n->params+2u,(uint16_t)count);
    if(count!=0u) ox_list(g,n,name,n->params+offset,count,0);
}
static uint16_t ox_opcode(const char* name) {
    static const char* const ops[]={"","Conv","Add","Mul","Div","Erf","HardSigmoid",
        "BatchNormalization","ReduceMean","Relu","AveragePool","Squeeze","Transpose",
        "Unsqueeze","MatMul","Softmax","Reshape","Concat","ConvTranspose","MaxPool",
        "Resize","Sigmoid","Sub","Sqrt","Pow","Slice"};
    uint16_t i; for(i=1u;i<26u;++i) if(strcmp(name,ops[i])==0) return i;
    if(strcmp(name,"GlobalAveragePool")==0) return 8u;
    return 0u;
}
static void ox_slice_params(ox_graph* g,ox_node* n,ox_value** in) {
    uint32_t count,i,j;
    if(n->count==1u) {
        ox_attr* a=ox_attr_find(n,"starts");
        OX_REQUIRE(g,a!=NULL && a->count!=0u && ox_attr_find(n,"ends")!=NULL,"ONNX Slice controls missing");
        count=a->count;
        ox_list(g,n,"starts",n->params+4u,count,0); ox_list(g,n,"ends",n->params+36u,count,0);
        ox_list(g,n,"steps",n->params+100u,count,1);
        if(ox_attr_find(n,"axes")!=NULL) ox_list(g,n,"axes",n->params+68u,count,0);
        else for(i=0u;i<count;++i) ox_put32(n->params+68u+i*4u,i);
    } else {
        OX_REQUIRE(g,n->count>=3u && in[1]->metadata && in[2]->metadata,"ONNX Slice constant controls");
        count=in[1]->control_count;
        OX_REQUIRE(g,count!=0u && count==in[2]->control_count,"ONNX Slice controls mismatch");
        for(j=0u;j<4u;++j) for(i=0u;i<count;++i) {
            int64_t x=j==2u?(int64_t)i:1;
            if(j+1u<n->count) {
                OX_REQUIRE(g,in[j+1u]->metadata && in[j+1u]->control_count==count,"ONNX Slice controls invalid");
                x=in[j+1u]->controls[i];
            }
            if(x>INT32_MAX) x=INT32_MAX;
            if(x<INT32_MIN) x=INT32_MIN;
            ox_put32(n->params+4u+j*32u+i*4u,(uint32_t)(int32_t)x);
        }
    }
    ox_put16(n->params+2u,(uint16_t)count); n->param_size=136u; n->lowered_count=1u;
}
static void ox_params(ox_graph* g,ox_node* n,ox_value** in) {
    uint16_t op=n->opcode; uint8_t* p=n->params;
    memset(p,0,sizeof(n->params)); n->param_size=0u; n->lowered_count=n->count; ox_put16(p,1u);
    if(op==1u || op==18u || op==10u || op==19u) {
        ox_attr* pad=ox_attr_find(n,"auto_pad"); uint32_t j;
        n->param_size=64u;
        OX_REQUIRE(g,ox_attr_find(n,"kernel_shape")!=NULL,"ONNX kernel_shape required");
        ox_list(g,n,"kernel_shape",p+8u,2u,0); ox_list(g,n,"strides",p+16u,2u,1);
        if(op==1u || op==18u) {
            ox_put16(p+2u,2u); ox_put32(p+4u,(uint32_t)ox_int(g,n,"group",1));
            ox_list(g,n,"dilations",p+24u,2u,1); ox_list(g,n,"pads",p+32u,4u,0);
            if(op==18u) {
                ox_attr* a=ox_attr_find(n,"output_padding");
                OX_REQUIRE(g,ox_attr_find(n,"output_shape")==NULL,"ONNX ConvTranspose output_shape unsupported");
                OX_REQUIRE(g,a==NULL || (a->count==2u && a->ints[0]==0 && a->ints[1]==0),
                           "ONNX ConvTranspose output_padding unsupported");
            }
        } else {
            ox_attr* a=ox_attr_find(n,"dilations");
            ox_put32(p+4u,2u); ox_list(g,n,"pads",p+24u,4u,0);
            ox_put32(p+40u,(uint32_t)ox_int(g,n,"ceil_mode",0));
            ox_put32(p+44u,(uint32_t)ox_int(g,n,"count_include_pad",0));
            OX_REQUIRE(g,a==NULL || (a->count==2u && a->ints[0]==1 && a->ints[1]==1),
                       "ONNX pool dilation unsupported");
        }
        if(pad!=NULL && strcmp(pad->string,"NOTSET")!=0) {
            OX_REQUIRE(g,pad->type==3u && strcmp(pad->string,"SAME_UPPER")==0 &&
                           lwm_read_i32(p+16u)==1 && lwm_read_i32(p+20u)==1 &&
                           op!=18u,"ONNX dynamic auto_pad unsupported");
            for(j=0u;j<2u;++j) {
                int64_t total=(int64_t)(lwm_read_i32(p+8u+j*4u)-1)*
                    (op==1u?lwm_read_i32(p+24u+j*4u):1);
                uint32_t off=op==1u?32u:24u;
                OX_REQUIRE(g,total>=0 && total<=INT32_MAX,"ONNX padding overflow");
                ox_put32(p+off+j*4u,(uint32_t)(total/2));
                ox_put32(p+off+8u+j*4u,(uint32_t)(total-total/2));
            }
        }
    } else if(op==6u) {
        n->param_size=16u; ox_float(p+4u,ox_fattr(g,n,"alpha",0.2f)); ox_float(p+8u,ox_fattr(g,n,"beta",0.5f));
    } else if(op==7u) {
        n->param_size=24u; ox_float(p+4u,ox_fattr(g,n,"epsilon",1e-5f)); ox_float(p+8u,ox_fattr(g,n,"momentum",0.9f));
        ox_put32(p+12u,(uint32_t)ox_int(g,n,"training_mode",0));
    } else if(op==8u) {
        n->param_size=48u;
        if(strcmp(n->op,"GlobalAveragePool")==0) {
            ox_put16(p+2u,2u); ox_put32(p+4u,1u); ox_put32(p+12u,2u); ox_put32(p+16u,3u);
        } else {
            ox_axes(g,n,"axes",12u); ox_put32(p+4u,(uint32_t)ox_int(g,n,"keepdims",1));
            ox_put32(p+8u,(uint32_t)ox_int(g,n,"noop_with_empty_axes",0));
        }
    } else if(op==11u || op==12u || op==13u) {
        n->param_size=40u; ox_axes(g,n,op==12u?"perm":"axes",4u);
    } else if(op==15u || op==17u) {
        n->param_size=16u; ox_put32(p+4u,(uint32_t)ox_int(g,n,"axis",1));
    } else if(op==20u) {
        ox_attr* mode=ox_attr_find(n,"mode");
        ox_attr* coord=ox_attr_find(n,"coordinate_transformation_mode");
        ox_attr* nearest=ox_attr_find(n,"nearest_mode"); uint32_t j;
        OX_REQUIRE(g,n->count==3u && in[1]->constant && in[1]->bytes==0u &&
                       in[2]->constant && in[2]->shape.dtype==LW_DTYPE_F32 && in[2]->bytes==16u &&
                       mode!=NULL && strcmp(mode->string,"nearest")==0 &&
                       coord!=NULL && strcmp(coord->string,"asymmetric")==0 &&
                       nearest!=NULL && strcmp(nearest->string,"floor")==0,"ONNX Resize pattern unsupported");
        n->param_size=32u; n->lowered_count=1u; ox_put16(p+2u,4u);
        for(j=0u;j<4u;++j) memcpy(p+4u+j*4u,in[2]->raw+j*4u,4u);
    } else if(op==25u) ox_slice_params(g,n,in);
    else if(op==16u) n->lowered_count=1u;
}
static void ox_numeric_shape(ox_graph* g,ox_node* n,ox_value** in,ox_value* out) {
    uint8_t storage[144]={0},node[72]={0};
    lw_runtime_tensor tensors[9]; lw_model model; lw_session session;
    uint32_t rank=in[0]->shape.rank,i;
    memset(&model,0,sizeof(model)); memset(&session,0,sizeof(session)); memset(tensors,0,sizeof(tensors));
    for(i=0u;i<n->lowered_count;++i) tensors[i]=in[i]->shape;
    if(n->opcode==2u || n->opcode==3u || n->opcode==4u || n->opcode==14u || n->opcode==22u || n->opcode==24u)
        rank=rank>in[1]->shape.rank?rank:in[1]->shape.rank;
    if(n->opcode==11u) {
        uint32_t axes=lwm_read_u16(n->params+2u);
        if(axes!=0u) { OX_REQUIRE(g,axes<=rank,"ONNX Squeeze rank"); rank-=axes; }
        else { rank=0u; for(i=0u;i<in[0]->shape.rank;++i) if(in[0]->shape.dimensions[i]!=1) ++rank; }
    } else if(n->opcode==13u) rank+=lwm_read_u16(n->params+2u);
    else if(n->opcode==8u && lwm_read_u32(n->params+4u)==0u) {
        uint32_t axes=lwm_read_u16(n->params+2u);
        OX_REQUIRE(g,axes<=rank,"ONNX ReduceMean rank"); rank=axes==0u?0u:rank-axes;
    } else if(n->opcode==16u) {
        uint64_t elements=1u,known=1u; int32_t missing=-1;
        OX_REQUIRE(g,n->count==2u && in[1]->metadata && in[1]->control_count!=0u,"ONNX Reshape controls");
        rank=in[1]->control_count;
        for(i=0u;i<in[0]->shape.rank;++i) {
            OX_REQUIRE(g,elements<=UINT64_MAX/(uint32_t)in[0]->shape.dimensions[i],"ONNX Reshape overflow");
            elements*=(uint32_t)in[0]->shape.dimensions[i];
        }
        for(i=0u;i<rank;++i) {
            int64_t d=in[1]->controls[i];
            if(d==0) { OX_REQUIRE(g,i<in[0]->shape.rank,"ONNX Reshape copy axis"); d=in[0]->shape.dimensions[i]; }
            OX_REQUIRE(g,d>=-1 && d!=0 && d<=INT32_MAX,"ONNX Reshape dimension");
            tensors[8].dimensions[i]=(int32_t)d;
            if(d==-1) { OX_REQUIRE(g,missing==-1,"ONNX Reshape multiple -1"); missing=(int32_t)i; }
            else { OX_REQUIRE(g,known<=UINT64_MAX/(uint32_t)d,"ONNX Reshape overflow"); known*=(uint32_t)d; }
        }
        if(missing>=0) {
            OX_REQUIRE(g,elements%known==0u && elements/known<=INT32_MAX,"ONNX Reshape mismatch");
            tensors[8].dimensions[missing]=(int32_t)(elements/known);
        }
    }
    OX_REQUIRE(g,rank<=8u,"ONNX resolved rank budget");
    tensors[8].dtype=in[0]->shape.dtype; tensors[8].rank=rank;
    if(n->opcode!=16u) for(i=0u;i<rank;++i) tensors[8].dimensions[i]=-1;
    model.bytes=storage; session.model=&model; session.tensors=tensors;
    ox_put16(node,n->opcode); ox_put16(node+2u,(uint16_t)n->lowered_count); ox_put16(node+4u,1u);
    for(i=0u;i<n->lowered_count;++i) ox_put32(node+8u+i*4u,i);
    ox_put32(node+40u,8u);
    if(n->param_size!=0u) { memcpy(storage+8u,n->params,n->param_size); ox_put64(node+56u,8u); }
    g->status=lw_resolve_import_node(&session,node,OX_MAX_BYTES,g->error);
    if(g->status!=LW_STATUS_OK) longjmp(*g->escape,1);
    out->shape=tensors[8]; out->ready=1u; out->metadata=0u;
}
static void ox_metadata(ox_graph* g,ox_node* n,ox_value** in,ox_value* out) {
    uint32_t i,j,count=0u; out->metadata=1u; out->ready=1u;
    if(strcmp(n->op,"Shape")==0) {
        out->control_count=in[0]->shape.rank;
        for(i=0u;i<out->control_count;++i) out->controls[i]=in[0]->shape.dimensions[i];
        out->shape.dtype=LW_DTYPE_I64; out->shape.rank=1u;
        out->shape.dimensions[0]=(int32_t)out->control_count;
    } else if(strcmp(n->op,"Concat")==0) {
        OX_REQUIRE(g,ox_int(g,n,"axis",0)==0,"ONNX metadata Concat axis");
        for(i=0u;i<n->count;++i) {
            OX_REQUIRE(g,in[i]->metadata && in[i]->control_count<=8u-count,"ONNX metadata Concat budget");
            for(j=0u;j<in[i]->control_count;++j) out->controls[count++]=in[i]->controls[j];
        }
        out->control_count=count;
    } else if(strcmp(n->op,"Slice")==0) {
        int64_t start,end,step; int32_t len=(int32_t)in[0]->control_count;
        OX_REQUIRE(g,lwm_read_u16(n->params+2u)==1u && lwm_read_i32(n->params+68u)==0,"ONNX metadata Slice axes");
        start=lwm_read_i32(n->params+4u); end=lwm_read_i32(n->params+36u); step=lwm_read_i32(n->params+100u);
        OX_REQUIRE(g,step>0,"ONNX metadata Slice step");
        if(start<0) start+=len;
        if(end<0) end+=len;
        if(start<0) start=0;
        if(start>len) start=len;
        if(end<0) end=0;
        if(end>len) end=len;
        for(;start<end;start+=step) out->controls[count++]=in[0]->controls[start];
        out->control_count=count;
    } else if(strcmp(n->op,"Squeeze")==0 || strcmp(n->op,"Unsqueeze")==0) {
        out->control_count=in[0]->control_count; memcpy(out->controls,in[0]->controls,sizeof(out->controls));
    } else ox_bad(g,LW_STATUS_UNSUPPORTED,"ONNX metadata reaches unsupported operator");
    if(strcmp(n->op,"Shape")!=0) {
        ox_numeric_shape(g,n,in,out);
        out->metadata=1u;
    }
}
static void ox_check_attributes(ox_graph* g,ox_node* n) {
    const char* allowed=",";
    uint32_t i;
    switch(n->opcode) {
    case 1u: allowed=",auto_pad,kernel_shape,strides,dilations,pads,group,"; break;
    case 18u: allowed=",auto_pad,kernel_shape,strides,dilations,pads,group,output_padding,output_shape,"; break;
    case 6u: allowed=",alpha,beta,"; break;
    case 7u: allowed=",epsilon,momentum,training_mode,spatial,"; break;
    case 8u: allowed=strcmp(n->op,"GlobalAveragePool")==0?",":",axes,keepdims,noop_with_empty_axes,"; break;
    case 10u: case 19u: allowed=",auto_pad,kernel_shape,strides,pads,dilations,ceil_mode,count_include_pad,storage_order,"; break;
    case 11u: case 13u: allowed=",axes,"; break;
    case 12u: allowed=",perm,"; break;
    case 15u: case 17u: allowed=",axis,"; break;
    case 20u: allowed=",mode,coordinate_transformation_mode,nearest_mode,"; break;
    case 25u: allowed=",starts,ends,axes,steps,"; break;
    default: break;
    }
    for(i=0u;i<n->attr_count;++i) {
        char key[68]; (void)snprintf(key,sizeof(key),",%s,",n->attrs[i].name);
        OX_REQUIRE(g,strstr(allowed,key)!=NULL,"ONNX unsupported operator attribute");
    }
    if(n->opcode==7u) OX_REQUIRE(g,ox_int(g,n,"spatial",1)==1,"ONNX nonspatial BN unsupported");
    if(n->opcode==19u) OX_REQUIRE(g,ox_int(g,n,"storage_order",0)==0,"ONNX pool storage order unsupported");
}
static void ox_arity(ox_graph* g,ox_node* n) {
    uint32_t lo=1u,hi=1u;
    if(n->opcode==1u || n->opcode==18u) { lo=2u; hi=3u; }
    else if(n->opcode==2u || n->opcode==3u || n->opcode==4u || n->opcode==14u ||
            n->opcode==16u || n->opcode==22u || n->opcode==24u) lo=hi=2u;
    else if(n->opcode==7u) lo=hi=5u;
    else if(n->opcode==20u) lo=hi=3u;
    else if(n->opcode==17u) hi=8u;
    else if(n->opcode==25u) hi=5u;
    OX_REQUIRE(g,n->count>=lo && n->count<=hi,"ONNX operator arity invalid");
}
/* Shape-only passes at two representative sizes discover changing axes for
 * this PP-OCR subset. Runtime session creation resolves the actual size using
 * the same shape rules; this is not a general symbolic ONNX shape engine. */
static void ox_lower(ox_graph* g,unsigned pass) {
    uint32_t i,j; lw_runtime_tensor input_shape=g->values[g->input].shape;
    for(i=0u;i<g->value_count;++i) {
        ox_value* v=&g->values[i]; v->ready=v->constant;
        if(!v->constant) v->metadata=0u;
    }
    input_shape.dimensions[0]=1;
    for(i=2u;i<4u;++i) if(g->values[g->input].dynamic[i]==-1)
        input_shape.dimensions[i]=pass==0u?320:640;
    g->values[g->input].shape=input_shape; g->values[g->input].ready=1u;
    for(i=0u;i<g->node_count;++i) {
        ox_node* n=&g->nodes[i]; ox_value* in[8]; ox_value* out=&g->values[n->output];
        if(strcmp(n->op,"Identity")==0) {
            uint32_t root;
            OX_REQUIRE(g,n->count==1u && n->inputs[0]!=OX_NONE && n->attr_count==0u,"ONNX Identity arity");
            root=ox_root(g,n->inputs[0]);
            OX_REQUIRE(g,g->values[root].ready,"ONNX graph not topologically ordered");
            out->alias=root; out->ready=1u; n->emitted=0u; continue;
        }
        n->opcode=ox_opcode(n->op); ox_arity(g,n); ox_check_attributes(g,n);
        for(j=0u;j<n->count;++j) {
            OX_REQUIRE(g,n->inputs[j]!=OX_NONE,"ONNX omitted input unsupported");
            in[j]=&g->values[ox_root(g,n->inputs[j])];
            OX_REQUIRE(g,in[j]->ready,"ONNX graph not topologically ordered");
        }
        if(strcmp(n->op,"Shape")==0) {
            OX_REQUIRE(g,n->count==1u && n->attr_count==0u,"ONNX Shape contract");
            ox_metadata(g,n,in,out); n->emitted=0u; continue;
        }
        OX_REQUIRE(g,n->opcode!=0u,"ONNX operator unsupported"); ox_params(g,n,in);
        if(in[0]->metadata) { ox_metadata(g,n,in,out); n->emitted=0u; continue; }
        for(j=1u;j<n->lowered_count;++j)
            OX_REQUIRE(g,!in[j]->metadata,"ONNX shape controls reach numeric operator");
        ox_numeric_shape(g,n,in,out); n->emitted=1u;
        if(pass==0u) out->first_rank=out->shape.rank;
        OX_REQUIRE(g,out->first_rank==out->shape.rank,"ONNX varying tensor rank unsupported");
        for(j=0u;j<out->shape.rank;++j) {
            if(pass==0u) out->first[j]=out->shape.dimensions[j];
            out->dynamic[j]=pass==0u?out->first[j]:
                (out->first[j]==out->shape.dimensions[j]?out->first[j]:-1);
        }
        if(n->opcode==16u && pass!=0u) {
            uint32_t unresolved=0u;
            for(j=0u;j<out->shape.rank;++j) if(out->dynamic[j]==-1) ++unresolved;
            OX_REQUIRE(g,unresolved<=1u,"ONNX dynamic Reshape multiple varying axes");
        }
    }
}
static size_t ox_align(size_t n) { return (n+7u)&~(size_t)7u; }
static void ox_serialize(ox_graph* g,lw_model* model) {
    uint32_t i,j,tensors=0u,nodes=0u;
    size_t tensor_off,node_off,param_off,param_bytes=0u,weight_off,weight_bytes=0u,size,k;
    uint8_t* b; uint64_t hash=UINT64_C(14695981039346656037);
    g->output=ox_root(g,g->output);
    OX_REQUIRE(g,g->values[g->output].ready && !g->values[g->output].metadata,"ONNX output not numeric");
    OX_REQUIRE(g,g->declared_output.dtype==g->values[g->output].shape.dtype &&
                   g->declared_output.rank==g->values[g->output].shape.rank,
               "ONNX output type/rank does not match graph");
    for(i=0u;i<g->declared_output.rank;++i)
        OX_REQUIRE(g,g->declared_output.dimensions[i]<=0 ||
                       g->declared_output.dimensions[i]==g->values[g->output].shape.dimensions[i],
                   "ONNX output dimension does not match graph");
    g->values[g->input].used=1u; g->values[g->output].used=1u;
    for(i=0u;i<g->node_count;++i) if(g->nodes[i].emitted) {
        ox_node* n=&g->nodes[i]; ++nodes; param_bytes+=ox_align(n->param_size);
        g->values[n->output].used=1u;
        for(j=0u;j<n->lowered_count;++j) g->values[ox_root(g,n->inputs[j])].used=1u;
    }
    for(i=0u;i<g->value_count;++i) if(g->values[i].used) {
        g->values[i].index=tensors++;
        if(g->values[i].constant) weight_bytes+=ox_align(g->values[i].bytes);
    }
    tensor_off=176u; node_off=tensor_off+(size_t)tensors*80u;
    param_off=node_off+(size_t)nodes*72u; weight_off=param_off+param_bytes; size=weight_off+weight_bytes;
    OX_REQUIRE(g,size<=OX_MAX_BYTES,"ONNX normalized graph budget");
    b=(uint8_t*)calloc(size,1u);
    if(b==NULL) ox_bad(g,LW_STATUS_OUT_OF_MEMORY,"ONNX normalized graph allocation");
    memcpy(b,"LWM0",4u); ox_put16(b+6u,1u); ox_put32(b+8u,160u); ox_put32(b+12u,1u);
    ox_put32(b+16u,tensors); ox_put32(b+20u,nodes); ox_put32(b+24u,1u); ox_put32(b+28u,1u);
    ox_put64(b+32u,160u); ox_put64(b+40u,168u); ox_put64(b+48u,tensor_off);
    ox_put64(b+56u,node_off); ox_put64(b+64u,param_off); ox_put64(b+72u,param_bytes);
    ox_put64(b+80u,weight_off); ox_put64(b+96u,weight_off); ox_put64(b+104u,weight_bytes);
    ox_put64(b+112u,size);
    ox_put32(b+160u,g->values[g->input].index); ox_put32(b+168u,g->values[g->output].index);
    weight_bytes=0u;
    for(i=0u;i<g->value_count;++i) if(g->values[i].used) {
        ox_value* v=&g->values[i]; uint8_t* t=b+tensor_off+v->index*80u;
        ox_put32(t,v->shape.dtype); ox_put32(t+4u,v->shape.rank);
        for(j=0u;j<v->shape.rank;++j)
            ox_put32(t+8u+j*4u,(uint32_t)(v->constant?v->shape.dimensions[j]:v->dynamic[j]));
        ox_put32(t+40u,(v->constant?1u:0u)|(i==g->input?2u:0u)|(i==g->output?4u:0u)); ox_put64(t+64u,UINT64_MAX);
        if(v->constant && v->bytes!=0u) {
            ox_put64(t+48u,weight_off+weight_bytes); ox_put64(t+56u,v->bytes);
            memcpy(b+weight_off+weight_bytes,v->raw,v->bytes); weight_bytes+=ox_align(v->bytes);
        }
    }
    nodes=0u; param_bytes=0u;
    for(i=0u;i<g->node_count;++i) if(g->nodes[i].emitted) {
        ox_node* n=&g->nodes[i]; uint8_t* p=b+node_off+nodes++*72u;
        ox_put16(p,n->opcode); ox_put16(p+2u,(uint16_t)n->lowered_count); ox_put16(p+4u,1u);
        for(j=0u;j<n->lowered_count;++j) ox_put32(p+8u+j*4u,g->values[ox_root(g,n->inputs[j])].index);
        ox_put32(p+40u,g->values[n->output].index);
        if(n->param_size!=0u) {
            ox_put64(p+56u,param_off+param_bytes); ox_put32(p+64u,n->param_size);
            memcpy(b+param_off+param_bytes,n->params,n->param_size); param_bytes+=ox_align(n->param_size);
        }
    }
    for(k=0u;k<size;++k) { hash^=b[k]; hash*=UINT64_C(1099511628211); }
    ox_put64(b+128u,hash); free(model->bytes); model->bytes=b; model->byte_count=size;
}
lw_status lw_import_onnx(lw_model* model,lw_error* error) {
    ox_graph* g=(ox_graph*)calloc(1u,sizeof(*g)); lw_status status; uint32_t i;
    if(g==NULL) { lw_set_error(error,LW_STATUS_OUT_OF_MEMORY,"ONNX importer allocation"); return LW_STATUS_OUT_OF_MEMORY; }
    g->error=error; g->input=OX_NONE; g->output=OX_NONE; g->status=LW_STATUS_INVALID_FORMAT;
    g->escape=(jmp_buf*)malloc(sizeof(*g->escape));
    if(g->escape==NULL) {
        free(g); lw_set_error(error,LW_STATUS_OUT_OF_MEMORY,"ONNX error context allocation");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    if(setjmp(*g->escape)==0) {
        ox_reader r={model->bytes,model->bytes+model->byte_count}; unsigned f,w,graphs=0u;
        OX_REQUIRE(g,model->byte_count<=OX_MAX_BYTES,"ONNX file budget exceeded");
        while(ox_tag(g,&r,&f,&w)) {
            if(f==7u && w==2u) {
                ox_reader graph=ox_child(g,&r),scan=graph;
                unsigned sf,sw; uint32_t constants=0u,nodes=0u;
                OX_REQUIRE(g,++graphs==1u,"ONNX duplicate graph");
                while(ox_tag(g,&scan,&sf,&sw)) {
                    if(sf==1u) ++nodes;
                    if(sf==5u) ++constants;
                    OX_REQUIRE(g,nodes<=OX_MAX_NODES && constants<=OX_MAX_VALUES,"ONNX table budget");
                    ox_skip(g,&scan,sw);
                }
                OX_REQUIRE(g,nodes!=0u,"ONNX empty graph unsupported");
                g->node_capacity=nodes;
                g->value_capacity=nodes*9u+constants+2u;
                if(g->value_capacity>OX_MAX_VALUES) g->value_capacity=OX_MAX_VALUES;
                g->values=(ox_value*)calloc(g->value_capacity,sizeof(*g->values));
                g->nodes=(ox_node*)calloc(g->node_capacity,sizeof(*g->nodes));
                if(g->values==NULL || g->nodes==NULL) ox_bad(g,LW_STATUS_OUT_OF_MEMORY,"ONNX graph tables allocation");
                ox_parse_graph(g,graph);
            } else if(f==8u && w==2u) {
                ox_reader c=ox_child(g,&r); unsigned cf,cw; char domain[128]={0}; uint64_t version=0u;
                while(ox_tag(g,&c,&cf,&cw)) {
                    if(cf==1u && cw==2u) ox_string(g,&c,domain,sizeof(domain));
                    else if(cf==2u && cw==0u) version=ox_var(g,&c);
                    else ox_skip(g,&c,cw);
                }
                OX_REQUIRE(g,(domain[0]==0 || strcmp(domain,"ai.onnx")==0) && g->opset==0u &&
                               (version==7u || version==11u || version==14u),"ONNX standard opset 7/11/14 required");
                g->opset=(uint32_t)version;
            } else if(f==25u) ox_bad(g,LW_STATUS_UNSUPPORTED,"ONNX functions unsupported");
            else ox_skip(g,&r,w);
        }
        OX_REQUIRE(g,graphs==1u && g->opset!=0u && g->input!=OX_NONE && g->output!=OX_NONE,"ONNX missing graph contract");
        memcpy(g->values[g->input].dynamic,g->values[g->input].shape.dimensions,32u);
        g->values[g->input].dynamic[0]=1;
        ox_lower(g,0u); ox_lower(g,1u); ox_serialize(g,model);
        g->status=lw_validate_lwm_v0(model,error);
    }
    status=g->status;
    if(g->values!=NULL) for(i=0u;i<g->value_count;++i) { free(g->values[i].name); free(g->values[i].owned); }
    free(g->values); free(g->nodes); free(g->escape); free(g); return status;
}
