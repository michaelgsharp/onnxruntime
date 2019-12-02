// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Featurizers;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Microsoft.Win32.SafeHandles;
using static Microsoft.ML.Featurizers.CommonExtensions;

[assembly: LoadableClass(typeof(ImputationMarkerTransformer), null, typeof(SignatureLoadModel),
    ImputationMarkerTransformer.UserName, ImputationMarkerTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(ImputationMarkerTransformer), null, typeof(SignatureLoadRowMapper),
ImputationMarkerTransformer.UserName, ImputationMarkerTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(ImputationMarkerEntrypoint))]

namespace Microsoft.ML.Featurizers
{
    public static class ImputationMarkerExtensionClass
    {
        public static ImputationMarkerEstimator ImputationMarkerTransformer(this TransformsCatalog catalog, string outputColumnName, string inputColumnName = null /* Insert additional params here as needed*/, bool treatDefaultAsNull = false)
        {
            var options = new ImputationMarkerEstimator.Options
            {
                Columns = new ImputationMarkerEstimator.Column[] { new ImputationMarkerEstimator.Column() { Name = outputColumnName, Source = inputColumnName ?? outputColumnName } },
                TreatDefaultAsNull = treatDefaultAsNull,
                /* Codegen: add extra options here as needed */
            };

            return new ImputationMarkerEstimator(CatalogUtils.GetEnvironment(catalog), options);
        }

        public static ImputationMarkerEstimator ImputationMarkerTransformer(this TransformsCatalog catalog, InputOutputColumnPair[] columns /* Insert additional params here as needed*/, bool treatDefaultAsNull = false)
        {
            var options = new ImputationMarkerEstimator.Options
            {
                Columns = columns.Select(x => new ImputationMarkerEstimator.Column { Name = x.OutputColumnName, Source = x.InputColumnName ?? x.OutputColumnName }).ToArray(),
                TreatDefaultAsNull = treatDefaultAsNull,
                /* Codegen: add extra options here as needed */
            };

            return new ImputationMarkerEstimator(CatalogUtils.GetEnvironment(catalog), options);
        }
    }

    public class ImputationMarkerEstimator : IEstimator<ImputationMarkerTransformer>
    {
        private Options _options;
        private readonly IHost _host;

        /* Codegen: Add additional needed class members here */

        #region Options

        /* If not one to one need to change this */
        internal sealed class Column : OneToOneColumn
        {
            internal static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                return TryUnparseCore(sb);
            }
        }

        internal sealed class Options: TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition (optional form: name:src)",
                Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "If default value for the variable should be treated as null",
    			Name = "TreatDefaultAsNull", ShortName = "DefaultNull", SortOrder = 2)]
			public bool TreatDefaultAsNull = false;

            /* Codegen: Add additonal options as needed */
        }

        #endregion

        internal ImputationMarkerEstimator(IHostEnvironment env, Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(ImputationMarkerEstimator));
            Contracts.CheckNonEmpty(options.Columns, nameof(options.Columns));
            /* Codegen: Any other checks for options go here */

            _options = options;
        }

        public ImputationMarkerTransformer Fit(IDataView input)
        {
            return new ImputationMarkerTransformer(_host, input, _options);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            var columns = inputSchema.ToDictionary(x => x.Name);

            foreach (var column in _options.Columns)
            {
                var inputColumn = columns[column.Source];

                if (!ImputationMarkerTransformer.TypedColumn.IsColumnTypeSupported(inputColumn.ItemType.RawType))
                    throw new InvalidOperationException($"Type {inputColumn.ItemType.RawType.ToString()} for column {column.Name} not a supported type.");

                /* Codegen: Do correct schema mapping here */

            }
            return new SchemaShape(columns.Values);
        }
    }

    public sealed class ImputationMarkerTransformer : RowToRowTransformerBase, IDisposable
    {
        #region Class data members

        internal const string Summary = ""; /* Insert summary here */
        internal const string UserName = "ImputationMarkerTransformer";
        internal const string ShortName = "ImputationMarkerTransformer";
        internal const string LoadName = "ImputationMarkerTransformer";
        internal const string LoaderSignature = "ImputationMarkerTransformer";

        private TypedColumn[] _columns;
        private ImputationMarkerEstimator.Options _options;

        #endregion

        internal ImputationMarkerTransformer(IHostEnvironment host, IDataView input, ImputationMarkerEstimator.Options options) :
            base(host.Register(nameof(ImputationMarkerTransformer)))
        {
            var schema = input.Schema;
            _options = options;

            _columns = options.Columns.Select(x => TypedColumn.CreateTypedColumn(x.Name, x.Source, schema[x.Source].Type.RawType.ToString(), this)).ToArray();
            foreach (var column in _columns)
            {
                column.CreateTransformerFromEstimator(input);
            }
        }

        // Factory method for SignatureLoadModel.
        internal ImputationMarkerTransformer(IHostEnvironment host, ModelLoadContext ctx) :
            base(host.Register(nameof(ImputationMarkerTransformer)))
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            /* Codegen: Edit this format as needed */
            // *** Binary format ***
            // int number of column pairs
            // for each column pair:
            //      string output column  name
            //      string input column name
            //      column type
            //      int length of c++ byte array
            //      byte array from c++

            var columnCount = ctx.Reader.ReadInt32();

            _options = new ImputationMarkerEstimator.Options();
            /* Codegen: Load any additional Options members here */

            _columns = new TypedColumn[columnCount];
            for (int i = 0; i < columnCount; i++)
            {
                _columns[i] = TypedColumn.CreateTypedColumn(ctx.Reader.ReadString(), ctx.Reader.ReadString(), ctx.Reader.ReadString(), this);

                // Load the C++ state and create the C++ transformer.
                var dataLength = ctx.Reader.ReadInt32();
                var data = ctx.Reader.ReadByteArray(dataLength);
                _columns[i].CreateTransformerFromSavedData(data);
            }
        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => new ImputationMarkerTransformer(env, ctx).MakeRowMapper(inputSchema);

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private static VersionInfo GetVersionInfo()
        {
            /* Codegen: Change these as needed */
            return new VersionInfo(
                modelSignature: "Enter 8 character long name here", /* Codegen: Enter * character name here */
                verWrittenCur: 0x00010001, /* Codegen: Update version numbers as necessary */
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ImputationMarkerTransformer).Assembly.FullName);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            /* Codegen: Edit this format as needed */
            // *** Binary format ***
            // int number of column pairs
            // for each column pair:
            //      string output column  name
            //      string input column name
            //      column type
            //      int length of c++ byte array
            //      byte array from c++

            ctx.Writer.Write(_columns.Count());

            /* Codegen: Write any _options members needed here */

            foreach (var column in _columns)
            {
                ctx.Writer.Write(column.Name);
                ctx.Writer.Write(column.Source);
                ctx.Writer.Write(column.Type);

                // Save C++ state
                var data = column.CreateTransformerSaveData();
                ctx.Writer.Write(data.Length);
                ctx.Writer.Write(data);
            }
        }

        public void Dispose()
        {
            foreach (var column in _columns)
            {
                column.Dispose();
            }
        }

        #region ColumnInfo

        #region BaseClass

        internal abstract class TypedColumn : IDisposable
        {
            internal readonly string Name;
            internal readonly string Source;
            internal readonly string Type;

            /* Codegen: Fill in supported types */
            private static readonly Type[] _supportedTypes = new Type[] { typeof(sbyte), typeof(short), typeof(int), typeof(long), typeof(byte), typeof(ushort), typeof(uint), typeof(ulong), typeof(float), typeof(double), typeof(bool), typeof(ReadOnlyMemory<char>) };

            /* Codegen: Any other needed members */

            internal TypedColumn(string name, string source, string type)
            {
                Name = name;
                Source = source;
                Type = type;
            }

            internal abstract void CreateTransformerFromEstimator(IDataView input);
            private protected abstract unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize);
            private protected abstract bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle);
            public abstract void Dispose();

            public abstract Type ReturnType();

            internal byte[] CreateTransformerSaveData()
            {

                var success = CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                using (var savedDataHandle = new SaveDataSafeHandle(buffer, bufferSize))
                {
                    byte[] savedData = new byte[bufferSize.ToInt32()];
                    Marshal.Copy(buffer, savedData, 0, savedData.Length);
                    return savedData;
                }
            }

            internal unsafe void CreateTransformerFromSavedData(byte[] data)
            {
                fixed (byte* rawData = data)
                {
                    IntPtr dataSize = new IntPtr(data.Count());
                    CreateTransformerFromSavedDataHelper(rawData, dataSize);
                }
            }

            internal static bool IsColumnTypeSupported(Type type)
            {
                return _supportedTypes.Contains(type);
            }

            internal static TypedColumn CreateTypedColumn(string name, string source, string type, ImputationMarkerTransformer parent)
            {
                if (type == typeof(sbyte).ToString())
				{
    				return new Int8TypedColumn(name, source, parent);
				}
				else if (type == typeof(short).ToString())
				{
    				return new Int16TypedColumn(name, source, parent);
				}
				else if (type == typeof(int).ToString())
				{
    				return new Int32TypedColumn(name, source, parent);
				}
				else if (type == typeof(long).ToString())
				{
    				return new Int64TypedColumn(name, source, parent);
				}
				else if (type == typeof(byte).ToString())
				{
    				return new UInt8TypedColumn(name, source, parent);
				}
				else if (type == typeof(ushort).ToString())
				{
    				return new UInt16TypedColumn(name, source, parent);
				}
				else if (type == typeof(uint).ToString())
				{
    				return new UInt32TypedColumn(name, source, parent);
				}
				else if (type == typeof(ulong).ToString())
				{
    				return new UInt64TypedColumn(name, source, parent);
				}
				else if (type == typeof(float).ToString())
				{
    				return new FloatTypedColumn(name, source, parent);
				}
				else if (type == typeof(double).ToString())
				{
    				return new DoubleTypedColumn(name, source, parent);
				}
				else if (type == typeof(bool).ToString())
				{
    				return new BoolTypedColumn(name, source, parent);
				}
				else if (type == typeof(ReadOnlyMemory<char>).ToString())
				{
    				return new ReadOnlyMemoryCharTypedColumn(name, source, parent);
				}

                throw new InvalidOperationException($"Column {name} has an unsupported type {type}.");
            }
        }

        internal abstract class TypedColumn<TSourceType, TOutputType> : TypedColumn
        {
            internal TypedColumn(string name, string source, string type) :
                base(name, source, type)
            {
            }

            internal abstract TOutputType Transform(TSourceType input);
            private protected abstract bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle);
            private protected abstract bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            private protected abstract bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle);
            private protected abstract bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle);
            private protected abstract bool FitHelper(TransformerEstimatorSafeHandle estimator, TSourceType input, out FitResult fitResult, out IntPtr errorHandle);
            private protected abstract bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle);
            private protected abstract bool IsTrainingComplete(TransformerEstimatorSafeHandle estimatorHandle);
            private protected TransformerEstimatorSafeHandle CreateTransformerFromEstimatorBase(IDataView input)
            {
                var success = CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                using (var estimatorHandle = new TransformerEstimatorSafeHandle(estimator, DestroyEstimatorHelper))
                {
                    if (!IsTrainingComplete(estimatorHandle))
                    {
                        var fitResult = FitResult.Continue;
                        while (fitResult != FitResult.Complete)
                        {
                            fitResult = FitResult.Continue;
                            using (var data = input.GetColumn<TSourceType>(Source).GetEnumerator())
                            {
                                while (fitResult == FitResult.Continue && data.MoveNext())
                                {
                                    {
                                        success = FitHelper(estimatorHandle, data.Current, out fitResult, out errorHandle);
                                        if (!success)
                                            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                                    }
                                }

                                success = CompleteTrainingHelper(estimatorHandle, out fitResult, out errorHandle);
                                if (!success)
                                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                            }
                        }
                    }
                    success = CreateTransformerFromEstimatorHelper(estimatorHandle, out IntPtr transformer, out errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    return new TransformerEstimatorSafeHandle(transformer, DestroyTransformerHelper);
                }
            }
        }

        #endregion
        
            #region Int8TypedColumn

            internal sealed class Int8TypedColumn : TypedColumn<sbyte, bool>
            {
                private TransformerEstimatorSafeHandle _transformerHandler;
                private ImputationMarkerTransformer _parent;
                internal Int8TypedColumn(string name, string source, ImputationMarkerTransformer parent) :
                    base(name, source, typeof(sbyte).ToString())
                {
                    _parent = parent;
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int8_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateEstimatorNative(/* Codegen: Add additional parameters here */ out IntPtr estimator, out IntPtr errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int8_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int8_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int8_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
                internal override void CreateTransformerFromEstimator(IDataView input)
                {
                    _transformerHandler = CreateTransformerFromEstimatorBase(input);
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int8_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
                private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
                {
                    var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                    if (!result)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int8_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, sbyte* input, out bool interopOutput, out IntPtr errorHandle);
                
                internal unsafe override bool Transform(sbyte input)
                {
                    sbyte* interopInput;
					if (_parent._options.TreatDefaultAsNull && input == default)
    					interopInput = null;
					else
    					interopInput = &input;

                    var success = TransformDataNative(_transformerHandler, interopInput, out bool interopOutput, out IntPtr errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    bool output = interopOutput;
                    return output;
                    
                }

                public override void Dispose()
                {
                    if (!_transformerHandler.IsClosed)
                        _transformerHandler.Dispose();
                }

                private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
                {
                    /* Codegen: do any extra checks/paramters here */
                    return CreateEstimatorNative(out estimator, out errorHandle);
                }

                private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                    CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

                private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                    DestroyEstimatorNative(estimator, out errorHandle);

                private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                    DestroyTransformerNative(transformer, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int8_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, sbyte* input, out FitResult fitResult, out IntPtr errorHandle);
                private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, sbyte input, out FitResult fitResult, out IntPtr errorHandle)
                {
                    sbyte* interopInput;
					if (_parent._options.TreatDefaultAsNull && input == default)
    					interopInput = null;
					else
    					interopInput = &input;

                    return FitNative(estimator, interopInput, out fitResult, out errorHandle);
                    
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int8_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle);
                private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle) =>
                        CompleteTrainingNative(estimator, out fitResult, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int8_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
                private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                    CreateTransformerSaveDataNative(_transformerHandler, out buffer, out bufferSize, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int8_IsTrainingComplete", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool IsTrainingCompleteNative(TransformerEstimatorSafeHandle transformer, out bool isTrainingComplete, out IntPtr errorHandle);
                private protected override bool IsTrainingComplete(TransformerEstimatorSafeHandle estimatorHandle)
                {
                    var success = IsTrainingCompleteNative(estimatorHandle, out bool isTrainingComplete, out IntPtr errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    return isTrainingComplete;
                }

                public override Type ReturnType()
                {
                    return typeof(bool);
                }
            }

            #endregion
            
            #region Int16TypedColumn

            internal sealed class Int16TypedColumn : TypedColumn<short, bool>
            {
                private TransformerEstimatorSafeHandle _transformerHandler;
                private ImputationMarkerTransformer _parent;
                internal Int16TypedColumn(string name, string source, ImputationMarkerTransformer parent) :
                    base(name, source, typeof(short).ToString())
                {
                    _parent = parent;
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int16_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateEstimatorNative(/* Codegen: Add additional parameters here */ out IntPtr estimator, out IntPtr errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int16_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int16_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int16_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
                internal override void CreateTransformerFromEstimator(IDataView input)
                {
                    _transformerHandler = CreateTransformerFromEstimatorBase(input);
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int16_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
                private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
                {
                    var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                    if (!result)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int16_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, short* input, out bool interopOutput, out IntPtr errorHandle);
                
                internal unsafe override bool Transform(short input)
                {
                    short* interopInput;
					if (_parent._options.TreatDefaultAsNull && input == default)
    					interopInput = null;
					else
    					interopInput = &input;

                    var success = TransformDataNative(_transformerHandler, interopInput, out bool interopOutput, out IntPtr errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    bool output = interopOutput;
                    return output;
                    
                }

                public override void Dispose()
                {
                    if (!_transformerHandler.IsClosed)
                        _transformerHandler.Dispose();
                }

                private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
                {
                    /* Codegen: do any extra checks/paramters here */
                    return CreateEstimatorNative(out estimator, out errorHandle);
                }

                private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                    CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

                private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                    DestroyEstimatorNative(estimator, out errorHandle);

                private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                    DestroyTransformerNative(transformer, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int16_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, short* input, out FitResult fitResult, out IntPtr errorHandle);
                private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, short input, out FitResult fitResult, out IntPtr errorHandle)
                {
                    short* interopInput;
					if (_parent._options.TreatDefaultAsNull && input == default)
    					interopInput = null;
					else
    					interopInput = &input;

                    return FitNative(estimator, interopInput, out fitResult, out errorHandle);
                    
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int16_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle);
                private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle) =>
                        CompleteTrainingNative(estimator, out fitResult, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int16_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
                private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                    CreateTransformerSaveDataNative(_transformerHandler, out buffer, out bufferSize, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int16_IsTrainingComplete", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool IsTrainingCompleteNative(TransformerEstimatorSafeHandle transformer, out bool isTrainingComplete, out IntPtr errorHandle);
                private protected override bool IsTrainingComplete(TransformerEstimatorSafeHandle estimatorHandle)
                {
                    var success = IsTrainingCompleteNative(estimatorHandle, out bool isTrainingComplete, out IntPtr errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    return isTrainingComplete;
                }

                public override Type ReturnType()
                {
                    return typeof(bool);
                }
            }

            #endregion
            
            #region Int32TypedColumn

            internal sealed class Int32TypedColumn : TypedColumn<int, bool>
            {
                private TransformerEstimatorSafeHandle _transformerHandler;
                private ImputationMarkerTransformer _parent;
                internal Int32TypedColumn(string name, string source, ImputationMarkerTransformer parent) :
                    base(name, source, typeof(int).ToString())
                {
                    _parent = parent;
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int32_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateEstimatorNative(/* Codegen: Add additional parameters here */ out IntPtr estimator, out IntPtr errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int32_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int32_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int32_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
                internal override void CreateTransformerFromEstimator(IDataView input)
                {
                    _transformerHandler = CreateTransformerFromEstimatorBase(input);
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int32_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
                private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
                {
                    var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                    if (!result)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int32_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, int* input, out bool interopOutput, out IntPtr errorHandle);
                
                internal unsafe override bool Transform(int input)
                {
                    int* interopInput;
					if (_parent._options.TreatDefaultAsNull && input == default)
    					interopInput = null;
					else
    					interopInput = &input;

                    var success = TransformDataNative(_transformerHandler, interopInput, out bool interopOutput, out IntPtr errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    bool output = interopOutput;
                    return output;
                    
                }

                public override void Dispose()
                {
                    if (!_transformerHandler.IsClosed)
                        _transformerHandler.Dispose();
                }

                private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
                {
                    /* Codegen: do any extra checks/paramters here */
                    return CreateEstimatorNative(out estimator, out errorHandle);
                }

                private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                    CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

                private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                    DestroyEstimatorNative(estimator, out errorHandle);

                private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                    DestroyTransformerNative(transformer, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int32_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, int* input, out FitResult fitResult, out IntPtr errorHandle);
                private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, int input, out FitResult fitResult, out IntPtr errorHandle)
                {
                    int* interopInput;
					if (_parent._options.TreatDefaultAsNull && input == default)
    					interopInput = null;
					else
    					interopInput = &input;

                    return FitNative(estimator, interopInput, out fitResult, out errorHandle);
                    
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int32_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle);
                private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle) =>
                        CompleteTrainingNative(estimator, out fitResult, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int32_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
                private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                    CreateTransformerSaveDataNative(_transformerHandler, out buffer, out bufferSize, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int32_IsTrainingComplete", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool IsTrainingCompleteNative(TransformerEstimatorSafeHandle transformer, out bool isTrainingComplete, out IntPtr errorHandle);
                private protected override bool IsTrainingComplete(TransformerEstimatorSafeHandle estimatorHandle)
                {
                    var success = IsTrainingCompleteNative(estimatorHandle, out bool isTrainingComplete, out IntPtr errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    return isTrainingComplete;
                }

                public override Type ReturnType()
                {
                    return typeof(bool);
                }
            }

            #endregion
            
            #region Int64TypedColumn

            internal sealed class Int64TypedColumn : TypedColumn<long, bool>
            {
                private TransformerEstimatorSafeHandle _transformerHandler;
                private ImputationMarkerTransformer _parent;
                internal Int64TypedColumn(string name, string source, ImputationMarkerTransformer parent) :
                    base(name, source, typeof(long).ToString())
                {
                    _parent = parent;
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int64_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateEstimatorNative(/* Codegen: Add additional parameters here */ out IntPtr estimator, out IntPtr errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int64_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int64_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int64_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
                internal override void CreateTransformerFromEstimator(IDataView input)
                {
                    _transformerHandler = CreateTransformerFromEstimatorBase(input);
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int64_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
                private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
                {
                    var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                    if (!result)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int64_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, long* input, out bool interopOutput, out IntPtr errorHandle);
                
                internal unsafe override bool Transform(long input)
                {
                    long* interopInput;
					if (_parent._options.TreatDefaultAsNull && input == default)
    					interopInput = null;
					else
    					interopInput = &input;

                    var success = TransformDataNative(_transformerHandler, interopInput, out bool interopOutput, out IntPtr errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    bool output = interopOutput;
                    return output;
                    
                }

                public override void Dispose()
                {
                    if (!_transformerHandler.IsClosed)
                        _transformerHandler.Dispose();
                }

                private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
                {
                    /* Codegen: do any extra checks/paramters here */
                    return CreateEstimatorNative(out estimator, out errorHandle);
                }

                private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                    CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

                private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                    DestroyEstimatorNative(estimator, out errorHandle);

                private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                    DestroyTransformerNative(transformer, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int64_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, long* input, out FitResult fitResult, out IntPtr errorHandle);
                private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, long input, out FitResult fitResult, out IntPtr errorHandle)
                {
                    long* interopInput;
					if (_parent._options.TreatDefaultAsNull && input == default)
    					interopInput = null;
					else
    					interopInput = &input;

                    return FitNative(estimator, interopInput, out fitResult, out errorHandle);
                    
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int64_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle);
                private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle) =>
                        CompleteTrainingNative(estimator, out fitResult, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int64_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
                private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                    CreateTransformerSaveDataNative(_transformerHandler, out buffer, out bufferSize, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_int64_IsTrainingComplete", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool IsTrainingCompleteNative(TransformerEstimatorSafeHandle transformer, out bool isTrainingComplete, out IntPtr errorHandle);
                private protected override bool IsTrainingComplete(TransformerEstimatorSafeHandle estimatorHandle)
                {
                    var success = IsTrainingCompleteNative(estimatorHandle, out bool isTrainingComplete, out IntPtr errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    return isTrainingComplete;
                }

                public override Type ReturnType()
                {
                    return typeof(bool);
                }
            }

            #endregion
            
            #region UInt8TypedColumn

            internal sealed class UInt8TypedColumn : TypedColumn<byte, bool>
            {
                private TransformerEstimatorSafeHandle _transformerHandler;
                private ImputationMarkerTransformer _parent;
                internal UInt8TypedColumn(string name, string source, ImputationMarkerTransformer parent) :
                    base(name, source, typeof(byte).ToString())
                {
                    _parent = parent;
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint8_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateEstimatorNative(/* Codegen: Add additional parameters here */ out IntPtr estimator, out IntPtr errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint8_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint8_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint8_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
                internal override void CreateTransformerFromEstimator(IDataView input)
                {
                    _transformerHandler = CreateTransformerFromEstimatorBase(input);
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint8_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
                private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
                {
                    var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                    if (!result)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint8_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, byte* input, out bool interopOutput, out IntPtr errorHandle);
                
                internal unsafe override bool Transform(byte input)
                {
                    byte* interopInput;
					if (_parent._options.TreatDefaultAsNull && input == default)
    					interopInput = null;
					else
    					interopInput = &input;

                    var success = TransformDataNative(_transformerHandler, interopInput, out bool interopOutput, out IntPtr errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    bool output = interopOutput;
                    return output;
                    
                }

                public override void Dispose()
                {
                    if (!_transformerHandler.IsClosed)
                        _transformerHandler.Dispose();
                }

                private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
                {
                    /* Codegen: do any extra checks/paramters here */
                    return CreateEstimatorNative(out estimator, out errorHandle);
                }

                private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                    CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

                private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                    DestroyEstimatorNative(estimator, out errorHandle);

                private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                    DestroyTransformerNative(transformer, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint8_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, byte* input, out FitResult fitResult, out IntPtr errorHandle);
                private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, byte input, out FitResult fitResult, out IntPtr errorHandle)
                {
                    byte* interopInput;
					if (_parent._options.TreatDefaultAsNull && input == default)
    					interopInput = null;
					else
    					interopInput = &input;

                    return FitNative(estimator, interopInput, out fitResult, out errorHandle);
                    
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint8_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle);
                private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle) =>
                        CompleteTrainingNative(estimator, out fitResult, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint8_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
                private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                    CreateTransformerSaveDataNative(_transformerHandler, out buffer, out bufferSize, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint8_IsTrainingComplete", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool IsTrainingCompleteNative(TransformerEstimatorSafeHandle transformer, out bool isTrainingComplete, out IntPtr errorHandle);
                private protected override bool IsTrainingComplete(TransformerEstimatorSafeHandle estimatorHandle)
                {
                    var success = IsTrainingCompleteNative(estimatorHandle, out bool isTrainingComplete, out IntPtr errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    return isTrainingComplete;
                }

                public override Type ReturnType()
                {
                    return typeof(bool);
                }
            }

            #endregion
            
            #region UInt16TypedColumn

            internal sealed class UInt16TypedColumn : TypedColumn<ushort, bool>
            {
                private TransformerEstimatorSafeHandle _transformerHandler;
                private ImputationMarkerTransformer _parent;
                internal UInt16TypedColumn(string name, string source, ImputationMarkerTransformer parent) :
                    base(name, source, typeof(ushort).ToString())
                {
                    _parent = parent;
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint16_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateEstimatorNative(/* Codegen: Add additional parameters here */ out IntPtr estimator, out IntPtr errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint16_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint16_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint16_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
                internal override void CreateTransformerFromEstimator(IDataView input)
                {
                    _transformerHandler = CreateTransformerFromEstimatorBase(input);
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint16_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
                private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
                {
                    var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                    if (!result)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint16_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, ushort* input, out bool interopOutput, out IntPtr errorHandle);
                
                internal unsafe override bool Transform(ushort input)
                {
                    ushort* interopInput;
					if (_parent._options.TreatDefaultAsNull && input == default)
    					interopInput = null;
					else
    					interopInput = &input;

                    var success = TransformDataNative(_transformerHandler, interopInput, out bool interopOutput, out IntPtr errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    bool output = interopOutput;
                    return output;
                    
                }

                public override void Dispose()
                {
                    if (!_transformerHandler.IsClosed)
                        _transformerHandler.Dispose();
                }

                private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
                {
                    /* Codegen: do any extra checks/paramters here */
                    return CreateEstimatorNative(out estimator, out errorHandle);
                }

                private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                    CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

                private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                    DestroyEstimatorNative(estimator, out errorHandle);

                private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                    DestroyTransformerNative(transformer, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint16_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, ushort* input, out FitResult fitResult, out IntPtr errorHandle);
                private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, ushort input, out FitResult fitResult, out IntPtr errorHandle)
                {
                    ushort* interopInput;
					if (_parent._options.TreatDefaultAsNull && input == default)
    					interopInput = null;
					else
    					interopInput = &input;

                    return FitNative(estimator, interopInput, out fitResult, out errorHandle);
                    
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint16_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle);
                private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle) =>
                        CompleteTrainingNative(estimator, out fitResult, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint16_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
                private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                    CreateTransformerSaveDataNative(_transformerHandler, out buffer, out bufferSize, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint16_IsTrainingComplete", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool IsTrainingCompleteNative(TransformerEstimatorSafeHandle transformer, out bool isTrainingComplete, out IntPtr errorHandle);
                private protected override bool IsTrainingComplete(TransformerEstimatorSafeHandle estimatorHandle)
                {
                    var success = IsTrainingCompleteNative(estimatorHandle, out bool isTrainingComplete, out IntPtr errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    return isTrainingComplete;
                }

                public override Type ReturnType()
                {
                    return typeof(bool);
                }
            }

            #endregion
            
            #region UInt32TypedColumn

            internal sealed class UInt32TypedColumn : TypedColumn<uint, bool>
            {
                private TransformerEstimatorSafeHandle _transformerHandler;
                private ImputationMarkerTransformer _parent;
                internal UInt32TypedColumn(string name, string source, ImputationMarkerTransformer parent) :
                    base(name, source, typeof(uint).ToString())
                {
                    _parent = parent;
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint32_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateEstimatorNative(/* Codegen: Add additional parameters here */ out IntPtr estimator, out IntPtr errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint32_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint32_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint32_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
                internal override void CreateTransformerFromEstimator(IDataView input)
                {
                    _transformerHandler = CreateTransformerFromEstimatorBase(input);
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint32_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
                private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
                {
                    var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                    if (!result)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint32_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, uint* input, out bool interopOutput, out IntPtr errorHandle);
                
                internal unsafe override bool Transform(uint input)
                {
                    uint* interopInput;
					if (_parent._options.TreatDefaultAsNull && input == default)
    					interopInput = null;
					else
    					interopInput = &input;

                    var success = TransformDataNative(_transformerHandler, interopInput, out bool interopOutput, out IntPtr errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    bool output = interopOutput;
                    return output;
                    
                }

                public override void Dispose()
                {
                    if (!_transformerHandler.IsClosed)
                        _transformerHandler.Dispose();
                }

                private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
                {
                    /* Codegen: do any extra checks/paramters here */
                    return CreateEstimatorNative(out estimator, out errorHandle);
                }

                private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                    CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

                private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                    DestroyEstimatorNative(estimator, out errorHandle);

                private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                    DestroyTransformerNative(transformer, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint32_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, uint* input, out FitResult fitResult, out IntPtr errorHandle);
                private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, uint input, out FitResult fitResult, out IntPtr errorHandle)
                {
                    uint* interopInput;
					if (_parent._options.TreatDefaultAsNull && input == default)
    					interopInput = null;
					else
    					interopInput = &input;

                    return FitNative(estimator, interopInput, out fitResult, out errorHandle);
                    
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint32_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle);
                private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle) =>
                        CompleteTrainingNative(estimator, out fitResult, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint32_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
                private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                    CreateTransformerSaveDataNative(_transformerHandler, out buffer, out bufferSize, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint32_IsTrainingComplete", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool IsTrainingCompleteNative(TransformerEstimatorSafeHandle transformer, out bool isTrainingComplete, out IntPtr errorHandle);
                private protected override bool IsTrainingComplete(TransformerEstimatorSafeHandle estimatorHandle)
                {
                    var success = IsTrainingCompleteNative(estimatorHandle, out bool isTrainingComplete, out IntPtr errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    return isTrainingComplete;
                }

                public override Type ReturnType()
                {
                    return typeof(bool);
                }
            }

            #endregion
            
            #region UInt64TypedColumn

            internal sealed class UInt64TypedColumn : TypedColumn<ulong, bool>
            {
                private TransformerEstimatorSafeHandle _transformerHandler;
                private ImputationMarkerTransformer _parent;
                internal UInt64TypedColumn(string name, string source, ImputationMarkerTransformer parent) :
                    base(name, source, typeof(ulong).ToString())
                {
                    _parent = parent;
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint64_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateEstimatorNative(/* Codegen: Add additional parameters here */ out IntPtr estimator, out IntPtr errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint64_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint64_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint64_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
                internal override void CreateTransformerFromEstimator(IDataView input)
                {
                    _transformerHandler = CreateTransformerFromEstimatorBase(input);
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint64_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
                private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
                {
                    var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                    if (!result)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint64_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, ulong* input, out bool interopOutput, out IntPtr errorHandle);
                
                internal unsafe override bool Transform(ulong input)
                {
                    ulong* interopInput;
					if (_parent._options.TreatDefaultAsNull && input == default)
    					interopInput = null;
					else
    					interopInput = &input;

                    var success = TransformDataNative(_transformerHandler, interopInput, out bool interopOutput, out IntPtr errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    bool output = interopOutput;
                    return output;
                    
                }

                public override void Dispose()
                {
                    if (!_transformerHandler.IsClosed)
                        _transformerHandler.Dispose();
                }

                private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
                {
                    /* Codegen: do any extra checks/paramters here */
                    return CreateEstimatorNative(out estimator, out errorHandle);
                }

                private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                    CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

                private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                    DestroyEstimatorNative(estimator, out errorHandle);

                private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                    DestroyTransformerNative(transformer, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint64_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, ulong* input, out FitResult fitResult, out IntPtr errorHandle);
                private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, ulong input, out FitResult fitResult, out IntPtr errorHandle)
                {
                    ulong* interopInput;
					if (_parent._options.TreatDefaultAsNull && input == default)
    					interopInput = null;
					else
    					interopInput = &input;

                    return FitNative(estimator, interopInput, out fitResult, out errorHandle);
                    
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint64_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle);
                private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle) =>
                        CompleteTrainingNative(estimator, out fitResult, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint64_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
                private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                    CreateTransformerSaveDataNative(_transformerHandler, out buffer, out bufferSize, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_uint64_IsTrainingComplete", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool IsTrainingCompleteNative(TransformerEstimatorSafeHandle transformer, out bool isTrainingComplete, out IntPtr errorHandle);
                private protected override bool IsTrainingComplete(TransformerEstimatorSafeHandle estimatorHandle)
                {
                    var success = IsTrainingCompleteNative(estimatorHandle, out bool isTrainingComplete, out IntPtr errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    return isTrainingComplete;
                }

                public override Type ReturnType()
                {
                    return typeof(bool);
                }
            }

            #endregion
            
            #region FloatTypedColumn

            internal sealed class FloatTypedColumn : TypedColumn<float, bool>
            {
                private TransformerEstimatorSafeHandle _transformerHandler;
                private ImputationMarkerTransformer _parent;
                internal FloatTypedColumn(string name, string source, ImputationMarkerTransformer parent) :
                    base(name, source, typeof(float).ToString())
                {
                    _parent = parent;
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_float_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateEstimatorNative(/* Codegen: Add additional parameters here */ out IntPtr estimator, out IntPtr errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_float_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_float_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_float_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
                internal override void CreateTransformerFromEstimator(IDataView input)
                {
                    _transformerHandler = CreateTransformerFromEstimatorBase(input);
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_float_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
                private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
                {
                    var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                    if (!result)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_float_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, float* input, out bool interopOutput, out IntPtr errorHandle);
                
                internal unsafe override bool Transform(float input)
                {
                    float* interopInput = &input;
                    var success = TransformDataNative(_transformerHandler, interopInput, out bool interopOutput, out IntPtr errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    bool output = interopOutput;
                    return output;
                    
                }

                public override void Dispose()
                {
                    if (!_transformerHandler.IsClosed)
                        _transformerHandler.Dispose();
                }

                private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
                {
                    /* Codegen: do any extra checks/paramters here */
                    return CreateEstimatorNative(out estimator, out errorHandle);
                }

                private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                    CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

                private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                    DestroyEstimatorNative(estimator, out errorHandle);

                private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                    DestroyTransformerNative(transformer, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_float_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, float* input, out FitResult fitResult, out IntPtr errorHandle);
                private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, float input, out FitResult fitResult, out IntPtr errorHandle)
                {
                    float* interopInput = &input;
                    return FitNative(estimator, interopInput, out fitResult, out errorHandle);
                    
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_float_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle);
                private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle) =>
                        CompleteTrainingNative(estimator, out fitResult, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_float_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
                private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                    CreateTransformerSaveDataNative(_transformerHandler, out buffer, out bufferSize, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_float_IsTrainingComplete", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool IsTrainingCompleteNative(TransformerEstimatorSafeHandle transformer, out bool isTrainingComplete, out IntPtr errorHandle);
                private protected override bool IsTrainingComplete(TransformerEstimatorSafeHandle estimatorHandle)
                {
                    var success = IsTrainingCompleteNative(estimatorHandle, out bool isTrainingComplete, out IntPtr errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    return isTrainingComplete;
                }

                public override Type ReturnType()
                {
                    return typeof(bool);
                }
            }

            #endregion
            
            #region DoubleTypedColumn

            internal sealed class DoubleTypedColumn : TypedColumn<double, bool>
            {
                private TransformerEstimatorSafeHandle _transformerHandler;
                private ImputationMarkerTransformer _parent;
                internal DoubleTypedColumn(string name, string source, ImputationMarkerTransformer parent) :
                    base(name, source, typeof(double).ToString())
                {
                    _parent = parent;
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_double_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateEstimatorNative(/* Codegen: Add additional parameters here */ out IntPtr estimator, out IntPtr errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_double_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_double_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_double_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
                internal override void CreateTransformerFromEstimator(IDataView input)
                {
                    _transformerHandler = CreateTransformerFromEstimatorBase(input);
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_double_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
                private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
                {
                    var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                    if (!result)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_double_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, double* input, out bool interopOutput, out IntPtr errorHandle);
                
                internal unsafe override bool Transform(double input)
                {
                    double* interopInput = &input;
                    var success = TransformDataNative(_transformerHandler, interopInput, out bool interopOutput, out IntPtr errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    bool output = interopOutput;
                    return output;
                    
                }

                public override void Dispose()
                {
                    if (!_transformerHandler.IsClosed)
                        _transformerHandler.Dispose();
                }

                private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
                {
                    /* Codegen: do any extra checks/paramters here */
                    return CreateEstimatorNative(out estimator, out errorHandle);
                }

                private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                    CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

                private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                    DestroyEstimatorNative(estimator, out errorHandle);

                private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                    DestroyTransformerNative(transformer, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_double_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, double* input, out FitResult fitResult, out IntPtr errorHandle);
                private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, double input, out FitResult fitResult, out IntPtr errorHandle)
                {
                    double* interopInput = &input;
                    return FitNative(estimator, interopInput, out fitResult, out errorHandle);
                    
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_double_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle);
                private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle) =>
                        CompleteTrainingNative(estimator, out fitResult, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_double_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
                private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                    CreateTransformerSaveDataNative(_transformerHandler, out buffer, out bufferSize, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_double_IsTrainingComplete", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool IsTrainingCompleteNative(TransformerEstimatorSafeHandle transformer, out bool isTrainingComplete, out IntPtr errorHandle);
                private protected override bool IsTrainingComplete(TransformerEstimatorSafeHandle estimatorHandle)
                {
                    var success = IsTrainingCompleteNative(estimatorHandle, out bool isTrainingComplete, out IntPtr errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    return isTrainingComplete;
                }

                public override Type ReturnType()
                {
                    return typeof(bool);
                }
            }

            #endregion
            
            #region BoolTypedColumn

            internal sealed class BoolTypedColumn : TypedColumn<bool, bool>
            {
                private TransformerEstimatorSafeHandle _transformerHandler;
                private ImputationMarkerTransformer _parent;
                internal BoolTypedColumn(string name, string source, ImputationMarkerTransformer parent) :
                    base(name, source, typeof(bool).ToString())
                {
                    _parent = parent;
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_bool_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateEstimatorNative(/* Codegen: Add additional parameters here */ out IntPtr estimator, out IntPtr errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_bool_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_bool_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_bool_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
                internal override void CreateTransformerFromEstimator(IDataView input)
                {
                    _transformerHandler = CreateTransformerFromEstimatorBase(input);
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_bool_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
                private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
                {
                    var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                    if (!result)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_bool_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, bool* input, out bool interopOutput, out IntPtr errorHandle);
                
                internal unsafe override bool Transform(bool input)
                {
                    bool* interopInput;
					if (_parent._options.TreatDefaultAsNull && input == default)
    					interopInput = null;
					else
    					interopInput = &input;

                    var success = TransformDataNative(_transformerHandler, interopInput, out bool interopOutput, out IntPtr errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    bool output = interopOutput;
                    return output;
                    
                }

                public override void Dispose()
                {
                    if (!_transformerHandler.IsClosed)
                        _transformerHandler.Dispose();
                }

                private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
                {
                    /* Codegen: do any extra checks/paramters here */
                    return CreateEstimatorNative(out estimator, out errorHandle);
                }

                private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                    CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

                private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                    DestroyEstimatorNative(estimator, out errorHandle);

                private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                    DestroyTransformerNative(transformer, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_bool_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, bool* input, out FitResult fitResult, out IntPtr errorHandle);
                private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, bool input, out FitResult fitResult, out IntPtr errorHandle)
                {
                    bool* interopInput;
					if (_parent._options.TreatDefaultAsNull && input == default)
    					interopInput = null;
					else
    					interopInput = &input;

                    return FitNative(estimator, interopInput, out fitResult, out errorHandle);
                    
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_bool_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle);
                private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle) =>
                        CompleteTrainingNative(estimator, out fitResult, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_bool_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
                private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                    CreateTransformerSaveDataNative(_transformerHandler, out buffer, out bufferSize, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_bool_IsTrainingComplete", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool IsTrainingCompleteNative(TransformerEstimatorSafeHandle transformer, out bool isTrainingComplete, out IntPtr errorHandle);
                private protected override bool IsTrainingComplete(TransformerEstimatorSafeHandle estimatorHandle)
                {
                    var success = IsTrainingCompleteNative(estimatorHandle, out bool isTrainingComplete, out IntPtr errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    return isTrainingComplete;
                }

                public override Type ReturnType()
                {
                    return typeof(bool);
                }
            }

            #endregion
            
            #region ReadOnlyMemoryCharTypedColumn

            internal sealed class ReadOnlyMemoryCharTypedColumn : TypedColumn<ReadOnlyMemory<char>, bool>
            {
                private TransformerEstimatorSafeHandle _transformerHandler;
                private ImputationMarkerTransformer _parent;
                internal ReadOnlyMemoryCharTypedColumn(string name, string source, ImputationMarkerTransformer parent) :
                    base(name, source, typeof(ReadOnlyMemory<char>).ToString())
                {
                    _parent = parent;
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_string_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateEstimatorNative(/* Codegen: Add additional parameters here */ out IntPtr estimator, out IntPtr errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_string_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_string_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_string_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
                internal override void CreateTransformerFromEstimator(IDataView input)
                {
                    _transformerHandler = CreateTransformerFromEstimatorBase(input);
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_string_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
                private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
                {
                    var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                    if (!result)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_string_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, byte* input, out bool interopOutput, out IntPtr errorHandle);
                
                internal unsafe override bool Transform(ReadOnlyMemory<char> input)
                {
                    var inputAsString = input.ToString();
					fixed (byte* interopInput = (string.IsNullOrEmpty(inputAsString) && _parent._options.TreatDefaultAsNull) ? null : Encoding.UTF8.GetBytes(inputAsString + char.MinValue))
					{

                    var success = TransformDataNative(_transformerHandler, interopInput, out bool interopOutput, out IntPtr errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    bool output = interopOutput;
                    return output;
                    }
                }

                public override void Dispose()
                {
                    if (!_transformerHandler.IsClosed)
                        _transformerHandler.Dispose();
                }

                private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
                {
                    /* Codegen: do any extra checks/paramters here */
                    return CreateEstimatorNative(out estimator, out errorHandle);
                }

                private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                    CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

                private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                    DestroyEstimatorNative(estimator, out errorHandle);

                private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                    DestroyTransformerNative(transformer, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_string_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, byte* input, out FitResult fitResult, out IntPtr errorHandle);
                private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, ReadOnlyMemory<char> input, out FitResult fitResult, out IntPtr errorHandle)
                {
                    var inputAsString = input.ToString();
					fixed (byte* interopInput = (string.IsNullOrEmpty(inputAsString) && _parent._options.TreatDefaultAsNull) ? null : Encoding.UTF8.GetBytes(inputAsString + char.MinValue))
					{

                    return FitNative(estimator, interopInput, out fitResult, out errorHandle);
                    }
                }

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_string_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle);
                private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle) =>
                        CompleteTrainingNative(estimator, out fitResult, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_string_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
                private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                    CreateTransformerSaveDataNative(_transformerHandler, out buffer, out bufferSize, out errorHandle);

                [DllImport("Featurizers", EntryPoint = "ImputationMarkerFeaturizer_string_IsTrainingComplete", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                private static extern bool IsTrainingCompleteNative(TransformerEstimatorSafeHandle transformer, out bool isTrainingComplete, out IntPtr errorHandle);
                private protected override bool IsTrainingComplete(TransformerEstimatorSafeHandle estimatorHandle)
                {
                    var success = IsTrainingCompleteNative(estimatorHandle, out bool isTrainingComplete, out IntPtr errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    return isTrainingComplete;
                }

                public override Type ReturnType()
                {
                    return typeof(bool);
                }
            }

            #endregion
            
        #endregion

        private sealed class Mapper : MapperBase
        {
            #region Class members

            private readonly ImputationMarkerTransformer _parent;
            /* Codegen: add any extra class members here */

            #endregion

            public Mapper(ImputationMarkerTransformer parent, DataViewSchema inputSchema) :
                base(parent.Host.Register(nameof(Mapper)), inputSchema, parent)
            {
                _parent = parent;
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                return _parent._columns.Select(x => new DataViewSchema.DetachedColumn(x.Name, ColumnTypeExtensions.PrimitiveTypeFromType(x.ReturnType()))).ToArray();
            }

            private Delegate MakeGetter<TSourceType, TOutputType>(DataViewRow input, int iinfo)
            {
                ValueGetter<TOutputType> result = (ref TOutputType dst) =>
                {
                    var inputColumn = input.Schema[_parent._columns[iinfo].Source];
                    var srcGetterScalar = input.GetGetter<TSourceType>(inputColumn);

                    TSourceType value = default;
                    srcGetterScalar(ref value);

                    dst = ((TypedColumn<TSourceType, TOutputType>)_parent._columns[iinfo]).Transform(value);

                };

                return result;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                Type inputType = input.Schema[_parent._columns[iinfo].Source].Type.RawType;
                Type outputType = _parent._columns[iinfo].ReturnType();

                return Utils.MarshalInvoke(MakeGetter<int, int>, new Type[] { inputType, outputType }, input, iinfo);
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                var active = new bool[InputSchema.Count];
                for (int i = 0; i < InputSchema.Count; i++)
                {
                    if (_parent._columns.Any(x => x.Source == InputSchema[i].Name))
                    {
                        active[i] = true;
                    }
                }

                return col => active[col];
            }

            private protected override void SaveModel(ModelSaveContext ctx) => _parent.SaveModel(ctx);
        }
    }

    internal static class ImputationMarkerEntrypoint
    {
        [TlcModule.EntryPoint(Name = "Transforms.ImputationMarker",
            Desc = ImputationMarkerTransformer.Summary,
            UserName = ImputationMarkerTransformer.UserName,
            ShortName = ImputationMarkerTransformer.ShortName)]
        public static CommonOutputs.TransformOutput ImputationMarker(IHostEnvironment env, ImputationMarkerEstimator.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, ImputationMarkerTransformer.ShortName, input);
            var xf = new ImputationMarkerEstimator(h, input).Fit(input.Data).Transform(input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }
    }
}
