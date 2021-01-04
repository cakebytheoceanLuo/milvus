package master

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	etcdkv "github.com/zilliztech/milvus-distributed/internal/kv/etcd"
	"github.com/zilliztech/milvus-distributed/internal/msgstream"
	"github.com/zilliztech/milvus-distributed/internal/proto/commonpb"
	pb "github.com/zilliztech/milvus-distributed/internal/proto/etcdpb"
	"github.com/zilliztech/milvus-distributed/internal/proto/internalpb"
	"github.com/zilliztech/milvus-distributed/internal/proto/schemapb"
	"github.com/zilliztech/milvus-distributed/internal/util/tsoutil"
	"github.com/zilliztech/milvus-distributed/internal/util/typeutil"
	"go.etcd.io/etcd/clientv3"
)

func TestSegmentManager_AssignSegment(t *testing.T) {
	ctx, cancelFunc := context.WithCancel(context.TODO())
	defer cancelFunc()

	Init()
	Params.TopicNum = 5
	Params.QueryNodeNum = 3
	Params.SegmentSize = 536870912 / 1024 / 1024
	Params.SegmentSizeFactor = 0.75
	Params.DefaultRecordSize = 1024
	Params.MinSegIDAssignCnt = 1048576 / 1024
	Params.SegIDAssignExpiration = 2000
	etcdAddress := Params.EtcdAddress
	cli, err := clientv3.New(clientv3.Config{Endpoints: []string{etcdAddress}})
	assert.Nil(t, err)
	rootPath := "/test/root"
	_, err = cli.Delete(ctx, rootPath, clientv3.WithPrefix())
	assert.Nil(t, err)

	kvBase := etcdkv.NewEtcdKV(cli, rootPath)
	defer kvBase.Close()
	mt, err := NewMetaTable(kvBase)
	assert.Nil(t, err)

	collName := "segmgr_test_coll"
	var collID int64 = 1001
	partitionTag := "test_part"
	schema := &schemapb.CollectionSchema{
		Name: collName,
		Fields: []*schemapb.FieldSchema{
			{FieldID: 1, Name: "f1", IsPrimaryKey: false, DataType: schemapb.DataType_INT32},
			{FieldID: 2, Name: "f2", IsPrimaryKey: false, DataType: schemapb.DataType_VECTOR_FLOAT, TypeParams: []*commonpb.KeyValuePair{
				{Key: "dim", Value: "128"},
			}},
		},
	}
	err = mt.AddCollection(&pb.CollectionMeta{
		ID:            collID,
		Schema:        schema,
		CreateTime:    0,
		SegmentIDs:    []UniqueID{},
		PartitionTags: []string{},
	})
	assert.Nil(t, err)
	err = mt.AddPartition(collID, partitionTag)
	assert.Nil(t, err)

	var cnt int64
	globalIDAllocator := func() (UniqueID, error) {
		val := atomic.AddInt64(&cnt, 1)
		return val, nil
	}
	globalTsoAllocator := func() (Timestamp, error) {
		val := atomic.AddInt64(&cnt, 1)
		phy := time.Now().UnixNano() / int64(time.Millisecond)
		ts := tsoutil.ComposeTS(phy, val)
		return ts, nil
	}
	syncWriteChan := make(chan *msgstream.TimeTickMsg)
	syncProxyChan := make(chan *msgstream.TimeTickMsg)

	segAssigner := NewSegmentAssigner(ctx, mt, globalTsoAllocator, syncProxyChan)
	mockScheduler := &MockFlushScheduler{}
	segManager, err := NewSegmentManager(ctx, mt, globalIDAllocator, globalTsoAllocator, syncWriteChan, mockScheduler, segAssigner)
	assert.Nil(t, err)

	segManager.Start()
	defer segManager.Close()
	sizePerRecord, err := typeutil.EstimateSizePerRecord(schema)
	assert.Nil(t, err)
	maxCount := uint32(Params.SegmentSize * 1024 * 1024 / float64(sizePerRecord))
	cases := []struct {
		Count         uint32
		ChannelID     int32
		Err           bool
		SameIDWith    int
		NotSameIDWith int
		ResultCount   int32
	}{
		{1000, 1, false, -1, -1, 1000},
		{2000, 0, false, 0, -1, 2000},
		{maxCount - 2999, 1, false, -1, 0, int32(maxCount - 2999)},
		{maxCount - 3000, 1, false, 0, -1, int32(maxCount - 3000)},
		{2000000000, 1, true, -1, -1, -1},
		{1000, 3, false, -1, 0, 1000},
		{maxCount, 2, false, -1, -1, int32(maxCount)},
	}

	var results = make([]*internalpb.SegIDAssignment, 0)
	for _, c := range cases {
		result, _ := segManager.AssignSegment([]*internalpb.SegIDRequest{{Count: c.Count, ChannelID: c.ChannelID, CollName: collName, PartitionTag: partitionTag}})
		results = append(results, result...)
		if c.Err {
			assert.EqualValues(t, commonpb.ErrorCode_UNEXPECTED_ERROR, result[0].Status.ErrorCode)
			continue
		}
		assert.EqualValues(t, commonpb.ErrorCode_SUCCESS, result[0].Status.ErrorCode)
		if c.SameIDWith != -1 {
			assert.EqualValues(t, result[0].SegID, results[c.SameIDWith].SegID)
		}
		if c.NotSameIDWith != -1 {
			assert.NotEqualValues(t, result[0].SegID, results[c.NotSameIDWith].SegID)
		}
		if c.ResultCount != -1 {
			assert.EqualValues(t, result[0].Count, c.ResultCount)
		}
	}

	time.Sleep(time.Duration(Params.SegIDAssignExpiration))
	timestamp, err := globalTsoAllocator()
	assert.Nil(t, err)
	err = mt.UpdateSegment(&pb.SegmentMeta{
		SegmentID:    results[0].SegID,
		CollectionID: collID,
		PartitionTag: partitionTag,
		ChannelStart: 0,
		ChannelEnd:   1,
		CloseTime:    timestamp,
		NumRows:      400000,
		MemSize:      500000,
	})
	assert.Nil(t, err)
	tsMsg := &msgstream.TimeTickMsg{
		BaseMsg: msgstream.BaseMsg{
			BeginTimestamp: timestamp, EndTimestamp: timestamp, HashValues: []uint32{},
		},
		TimeTickMsg: internalpb.TimeTickMsg{
			MsgType:   internalpb.MsgType_kTimeTick,
			PeerID:    1,
			Timestamp: timestamp,
		},
	}
	syncWriteChan <- tsMsg
	time.Sleep(300 * time.Millisecond)
	segMeta, err := mt.GetSegmentByID(results[0].SegID)
	assert.Nil(t, err)
	assert.NotEqualValues(t, 0, segMeta.CloseTime)
}